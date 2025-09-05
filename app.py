# modal_app.py
"""
SGLang kernel/dev lab on Modal with cost-safe profiling for GPT-OSS 20B.

Key funcs:
  - prewarm(model_id)                # hydrate weights on CPU (or tiny GPU) into /hf
  - build_kernels(ref)               # optional DeepGEMM (or other) precompile
  - serve()                          # SGLang HTTP server (OpenAI-compatible), optional API key
  - inference_test(prompt, ...)      # quick sanity + latency test (offline engine)
  - profile_inference(...)           # PyTorch CUDA profiler + NVTX trace (saved under /logs)
  - profile_step(...)                # short synthetic decode microbench for kernel work
  - endpoint_url()                   # fetch public URL of serve()

Env you’ll likely set:
  SGLANG_MODEL_ID=openai/gpt-oss-20b
  SGLANG_ATTENTION_BACKEND=triton|flashinfer|pytorch
  SGLANG_API_KEY=...                 # if you expose /v1 privately
  MODAL_GPU=L4|A100|H100             # default A100
  DEBUG_KERNELS=1                    # NVCC -G -lineinfo, CUDA_LAUNCH_BLOCKING=1
  USE_COMPUTE_SANITIZER=1            # wraps server with compute-sanitizer
"""

import os
import sys
import json
import time
import signal
import subprocess
from datetime import datetime

import modal
from modal import Image, Volume, gpu

# -------------------------
# App & configuration
# -------------------------
app = modal.App("sglang-kernel-lab")

# Your fork (editable) so you can tweak kernels quickly
SGLANG_GIT_URL = os.environ.get("SGLANG_GIT_URL", "https://github.com/jIab-b/sglang")
SGLANG_GIT_REF = os.environ.get("SGLANG_GIT_REF", "main")  # branch/tag/commit
DEFAULT_MODEL_ID = os.environ.get("SGLANG_MODEL_ID", "openai/gpt-oss-20b")  # gpt-oss-20b by default :contentReference[oaicite:2]{index=2}

# GPU type (dev on L4/L40S, finals on A100/H100)
MODAL_GPU = os.environ.get("MODAL_GPU", "A100").upper()
GPU_RESOURCE = {"A100": gpu.A100(), "H100": gpu.H100(), "L4": gpu.L4()}.get(MODAL_GPU, gpu.A100())

# Volumes: weights/datasets, compiler/runtime caches, logs/traces
hf_volume = Volume.from_name("hf-cache", create_if_missing=True)
kernels_volume = Volume.from_name("kernels-cache", create_if_missing=True)
logs_volume = Volume.from_name("sg-logs", create_if_missing=True)

# -------------------------
# Base image with CUDA + Python
# -------------------------
# We bring CUDA 12.8 devel so you get nvcc and low-level tooling, and install torch==2.8.* + cu128 wheels
# to avoid ABI mismatch (PyTorch publishes cu128 wheels for 2.8).
base_devel = (
    Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .run_commands([
        "sed -i 's|http://archive.ubuntu.com|http://au.archive.ubuntu.com|g' /etc/apt/sources.list",
        "apt-get update",
    ])
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
        "cmake",
        "gdb",
        "wget",
        "curl",
        "pciutils",
    )
    # Install torch/cu128 first to guarantee GPU wheels
    .uv_pip_install(
        "torch==2.8.*",
        "torchvision==0.21.*",
        extra_options=["--index-url", "https://download.pytorch.org/whl/cu128"],
    )
)

# Clone your sglang fork and install; also install deps we need.
# We set all caches to live on /hf and /kernels_cache (persisted volumes).
sglang_image = (
    base_devel
    .run_commands([f"git clone --depth 1 --branch {SGLANG_GIT_REF} {SGLANG_GIT_URL} /opt/sglang"])
    .env({
        # Hugging Face caches → /hf
        "HF_HOME": "/hf",
        "TRANSFORMERS_CACHE": "/hf/transformers",
        "HF_DATASETS_CACHE": "/hf/datasets",
        # Compiler/runtime caches → /kernels_cache
        "TORCHINDUCTOR_CACHE_DIR": "/kernels_cache/inductor",
        "TRITON_CACHE_DIR": "/kernels_cache/triton",
        "CUDA_CACHE_PATH": "/kernels_cache/nv",
        "XDG_CACHE_HOME": "/kernels_cache/xdg",
        # Stable CPU threads for more repeatable timings
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        # By default, avoid surprise DeepGEMM JIT on first request (can enable later)
        "SGL_ENABLE_JIT_DEEPGEMM": os.environ.get("SGL_ENABLE_JIT_DEEPGEMM", "0"),
    })
    .uv_pip_install(
        "/opt/sglang[all]",
        extra_options=["-e"],
    )
    .uv_pip_install(
        # Runtime/test deps (versions here are flexible; sglang pins its own)
        "transformers>=4.55.0",   # GPT-OSS model card recommends recent Transformers for chat template/harmony format :contentReference[oaicite:3]{index=3}
        "flashinfer-python>=0.2.8",
        "huggingface_hub>=0.23.0",
        "hf_transfer>=0.1.5",
        "requests>=2.32.0",
        "numpy",
        "packaging",
        # optional: PyTorch profiler tensorboard plugin writer
        "tensorboard",
    )
    # OPTIONAL (uncomment to bundle Nsight headless CLIs)
    .run_commands([
        "wget -qO- https://developer.nvidia.com/downloads/nsight-systems-2024-5-1-linux64 | tar -xj -C /opt",
        "ln -s /opt/nsight-systems-*/bin/nsys /usr/local/bin/nsys",
        "wget -qO- https://developer.nvidia.com/downloads/nsight-compute-2024-3-0-linux64 | tar -xJ -C /opt",
        "ln -s /opt/NsightCompute-*/nv-nsight-cu-cli /usr/local/bin/ncu",
    ])
)

# -------------------------
# Utilities
# -------------------------
def _ensure_dirs():
    os.makedirs("/logs/sglang", exist_ok=True)
    os.makedirs("/logs/sglang/crash_dumps", exist_ok=True)
    os.makedirs("/logs/sglang/traces", exist_ok=True)

def _apply_debug_env(env: dict) -> dict:
    # Kernel debug flags, Triton cache, CUDA sync for deterministic debugging
    if os.environ.get("DEBUG_KERNELS", "0") == "1":
        prepend = env.get("NVCC_PREPEND_FLAGS", "").strip()
        debug_flags = "-G -lineinfo --ptxas-options=-v"
        env["NVCC_PREPEND_FLAGS"] = f"{debug_flags} {prepend}".strip()
        env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        env.setdefault("TRITON_DEBUG", "1")
        env.setdefault("TRITON_CACHE_DIR", "/kernels_cache/triton")
    return env

def _maybe_wrap_with_compute_sanitizer(cmd: list[str]) -> list[str]:
    if os.environ.get("USE_COMPUTE_SANITIZER", "0") == "1":
        log_file = f"/logs/sglang/compute-sanitizer_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log"
        return [
            "compute-sanitizer",
            "--tool",
            os.environ.get("COMPUTE_SANITIZER_TOOL", "memcheck"),
            "--target-processes",
            "all",
            "--log-file",
            log_file,
            "--",
            *cmd,
        ]
    return cmd

def _build_server_args() -> list[str]:
    args = ["python", "-m", "sglang.launch_server"]
    model_id = os.environ.get("SGLANG_MODEL_ID", DEFAULT_MODEL_ID)
    if model_id:
        args += ["--model-path", model_id]

    # common runtime flags
    tp = os.environ.get("SGLANG_TENSOR_PARALLEL", "1")
    max_bs = os.environ.get("SGLANG_MAX_BATCH_SIZE", "32")  # unused currently, placeholder
    host = os.environ.get("SGLANG_HOST", "0.0.0.0")
    port = os.environ.get("SGLANG_PORT", "8000")
    args += ["--tensor-parallel-size", str(tp)]
    args += ["--host", host, "--port", str(port)]
    args += ["--download-dir", "/hf"]

    # attention/sampling backend toggles (useful for kernel diffs)
    attn = os.environ.get("SGLANG_ATTENTION_BACKEND", "").strip()
    if attn:
        args += ["--attention-backend", attn]
    samp = os.environ.get("SGLANG_SAMPLING_BACKEND", "").strip()
    if samp:
        args += ["--sampling-backend", samp]

    # Optional API key (server will enforce Authorization: Bearer <key>)
    api_key = os.environ.get("SGLANG_API_KEY", "").strip()
    if api_key:
        args += ["--api-key", api_key]  # supported in recent sglang builds :contentReference[oaicite:4]{index=4}

    # Logging & metrics
    log_level = os.environ.get("SGLANG_LOG_LEVEL", "info")
    args += ["--log-level", log_level]
    if os.environ.get("SGLANG_ENABLE_METRICS", "0") == "1":
        args += ["--enable-metrics"]
    args += ["--crash-dump-folder", "/logs/sglang/crash_dumps"]

    # pass-through extras (space-separated)
    extra = os.environ.get("SGLANG_EXTRA_ARGS", "").strip()
    if extra:
        args += extra.split()
    return args

# -------------------------
# Functions
# -------------------------

@app.function(
    image=sglang_image,
    gpu=None,  # CPU prewarm to avoid burning GPU time
    volumes={"/hf": hf_volume},
    timeout=30 * 60,
        max_containers=2,
)
def prewarm(model_id: str = DEFAULT_MODEL_ID) -> str:
    """
    Download tokenizer + weights into /hf, no GPU billed.
    """
    from huggingface_hub import snapshot_download
    target = f"/hf/models/{model_id.replace('/', '__')}"
    os.environ.setdefault("HF_HOME", "/hf")
    snapshot_download(
        repo_id=model_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        revision=os.environ.get("HF_REVISION", None),
        ignore_patterns=["*.msgpack", "*.pt", "*.bin.index.json"],  # small sanity filter
    )
    return target

@app.function(
    image=sglang_image,
    gpu=GPU_RESOURCE,
    volumes={"/hf": hf_volume, "/kernels_cache": kernels_volume, "/logs": logs_volume},
    timeout=24 * 60 * 60,
    max_containers=1,   # guardrail: one long-lived server at a time
)
@modal.web_server(port=8000)
def serve():
    """
    Start the SGLang HTTP server (OpenAI-compatible).
    Streams stdout to Modal logs and /logs/sglang/server_<run>.log.
    """
    _ensure_dirs()
    env = _apply_debug_env(os.environ.copy())

    # HF token passthrough (optional)
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    # Build & maybe wrap command
    cmd = _build_server_args()
    cmd = _maybe_wrap_with_compute_sanitizer(cmd)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    server_log_path = f"/logs/sglang/server_{run_id}.log"

    with open(server_log_path, "a", buffering=1) as logf:
        logf.write(f"\n===== Launch: {datetime.utcnow().isoformat()}Z =====\n")
        logf.write("CMD: " + " ".join(cmd) + "\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        def _term(signum, frame):
            try:
                proc.terminate()
            except Exception:
                pass

        signal.signal(signal.SIGTERM, _term)

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                # stream to Modal logs and also to file
                print(line, end="")
                try:
                    logf.write(line)
                except Exception:
                    pass
        finally:
            code = proc.wait()
            logf.write(f"\n===== Exit code: {code} at {datetime.utcnow().isoformat()}Z =====\n")
            logf.flush()

@app.function(
    image=sglang_image,
    gpu=GPU_RESOURCE,
    volumes={"/kernels_cache": kernels_volume, "/logs": logs_volume},
    timeout=30 * 60,
        max_containers=2,
)
def build_kernels(ref: str = "") -> str:
    """
    Optional kernel prebuild (e.g., DeepGEMM) with logs persisted.
    DEBUG_KERNELS=1 for NVCC debug flags. USE_COMPUTE_SANITIZER=1 to wrap commands at runtime (server only).
    """
    _ensure_dirs()
    env = _apply_debug_env(os.environ.copy())
    ref = ref or SGLANG_GIT_REF

    log_path = f"/logs/sglang/build_kernels_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log"
    code = 0
    with open(log_path, "a", buffering=1) as logf:
        logf.write(f"\n===== build_kernels start {datetime.utcnow().isoformat()}Z (ref={ref}) =====\n")
        try:
            # Try sglang's deep gemm compile entrypoint if available
            py = r"""
import sys
print("Attempting SGLang kernel/DeepGEMM precompile…")
try:
    from sglang import compile_deep_gemm
    compile_deep_gemm()
    print("compile_deep_gemm() invoked successfully.")
except Exception as e:
    print("compile_deep_gemm unavailable or failed:", e)
"""
            proc = subprocess.Popen(
                ["python", "-c", py],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                try:
                    logf.write(line)
                except Exception:
                    pass
            code = proc.wait()
        finally:
            logf.write(f"\n===== build_kernels exit {code} at {datetime.utcnow().isoformat()}Z =====\n")
    return json.dumps({"ref": ref, "exit_code": code, "log": log_path})

@app.function(
    image=sglang_image,
    gpu=GPU_RESOURCE,
    volumes={"/hf": hf_volume, "/kernels_cache": kernels_volume, "/logs": logs_volume},
    timeout=15 * 60,
        max_containers=2,
)
def inference_test(
    prompt: str = "Write a short haiku about GPUs.",
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    attention_backend: str = "",
) -> dict:
    """
    Load SGLang offline engine in-process and run a quick prompt.
    Returns latency and tokens/sec. Use this to sanity check gpt-oss-20b.
    """
    import torch
    import sglang as sgl  # offline Engine API is supported and avoids HTTP overhead :contentReference[oaicite:5]{index=5}

    if attention_backend:
        os.environ["SGLANG_ATTENTION_BACKEND"] = attention_backend

    # Engine will pull from /hf (already prewarmed)
    t0 = time.time()
    llm = sgl.Engine(model_path=model_id)  # dtype/attn are decided internally by SGLang
    t1 = time.time()

    prompts = [prompt]
    sampling_params = {"temperature": temperature, "max_new_tokens": max_new_tokens}

    torch.cuda.synchronize()
    t2 = time.time()
    outs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    t3 = time.time()

    # crude token count: rely on returned text length as proxy + max_new_tokens bound (engine doesn't expose tokens here)
    gen_text = outs[0]["text"]
    load_s = round(t1 - t0, 4)
    infer_s = round(t3 - t2, 4)
    llm.shutdown()

    return {
        "model": model_id,
        "load_seconds": load_s,
        "inference_seconds": infer_s,
        "response_preview": gen_text[:256],
    }

@app.function(
    image=sglang_image,
    gpu=GPU_RESOURCE,
    volumes={"/hf": hf_volume, "/kernels_cache": kernels_volume, "/logs": logs_volume},
    timeout=20 * 60,
    max_containers=1,
)
def profile_inference(
    model_id: str = DEFAULT_MODEL_ID,
    seq: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    attention_backend: str = "triton",
    trace_name: str = "prof_gptoss20b",
) -> dict:
    """
    Collect a CUDA kernel timeline using PyTorch Profiler + NVTX around a single prompt.
    Produces /logs/sglang/traces/<trace_name>.json (Chrome trace format) you can open in Perfetto/Chrome.

    If you prefer Nsight Systems/Compute, bundle their CLIs into the image (commented above)
    and wrap this function via `nsys profile ... modal run` externally.
    """
    import os
    import json as _json
    import torch
    import sglang as sgl

    os.environ["SGLANG_ATTENTION_BACKEND"] = attention_backend

    # Simple prompt sized to exercise prefill+decode at the given seq length.
    base = "You are a precise assistant. Summarize this text in 2 sentences:\n"
    filler = "The quick brown fox jumps over the lazy dog. " * ((seq // 9) + 1)
    prompt = (base + filler)[:seq]

    llm = sgl.Engine(model_path=model_id)

    # Mark the region with NVTX so external tools can zoom in.
    try:
        torch.cuda.nvtx.range_push("SGLANG_GENERATE")
    except Exception:
        pass

    # PyTorch CUDA profiler (kernel timeline)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_modules=True,
    ) as prof:
        outs = llm.generate([prompt], {"temperature": temperature, "max_new_tokens": max_new_tokens})

    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass

    llm.shutdown()

    # Save Chrome trace
    trace_path = f"/logs/sglang/traces/{trace_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    prof.export_chrome_trace(trace_path)

    return {
        "model": model_id,
        "trace_path": trace_path,
        "preview": outs[0]["text"][:160],
    }

@app.function(
    image=sglang_image,
    gpu=GPU_RESOURCE,
    volumes={"/hf": hf_volume, "/kernels_cache": kernels_volume, "/logs": logs_volume},
    timeout=10 * 60,
    max_containers=1,
)
def profile_step(
    seq: int = 4096,
    bsz: int = 1,
    steps: int = 20,
    attention_backend: str = "triton",
) -> dict:
    """
    Short microbench for kernel iteration. Keep this quick (<10s).
    """
    import time
    import torch
    import sglang as sgl

    os.environ["SGLANG_ATTENTION_BACKEND"] = attention_backend

    # synthetic decode-ish workload via repeated short generations
    llm = sgl.Engine(model_path=DEFAULT_MODEL_ID)
    prompt = "NVIDIA GPUs enable fast attention. Finish the sentence:"

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        _ = llm.generate([prompt], {"temperature": 0.0, "max_new_tokens": 16})
    torch.cuda.synchronize()
    dt = time.time() - t0
    llm.shutdown()

    return {"seconds": round(dt, 4), "seq": seq, "bsz": bsz, "steps": steps, "backend": attention_backend}

@app.function()
def endpoint_url() -> str:
    try:
        fn = modal.Function.from_name(app.name, "serve")
        url = fn.get_web_url()
        return url or ""
    except Exception as e:
        print(f"Error getting web URL for serve function: {e}")
        return ""

# -------------------------
# Local orchestration helper
# -------------------------
@app.local_entrypoint()
def main(
    do_prewarm: int = 1,
    do_build: int = 0,
    smoke_prompt: str = "hello from modal",
):
    """
    Convenience entrypoint for local dev:
      modal run modal_app.py::main --do-prewarm 1 --smoke-prompt "hi"
    """
    if do_prewarm:
        path = prewarm.remote(DEFAULT_MODEL_ID)
        print("Prewarmed:", path)

    if do_build:
        res = build_kernels.remote()
        print("Build kernels:", res)

    # quick end-to-end sanity via offline engine
    result = inference_test.remote(prompt=smoke_prompt, model_id=DEFAULT_MODEL_ID, max_new_tokens=64)
    print("Inference test:", result)

    # fetch (if deployed) URL for serve()
    try:
        url = endpoint_url.remote()
        print("Serve URL:", url)
    except Exception as e:
        print("Endpoint URL error:", e)
