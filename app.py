# modal_app.py
import os, signal, subprocess
from datetime import datetime
import modal
from modal import Image, Volume, gpu

app = modal.App("sglang-official-dev")

# volumes: models, jit/compile caches, logs
hf  = Volume.from_name("hf-cache", create_if_missing=True)
kc  = Volume.from_name("kernels-cache", create_if_missing=True)
log = Volume.from_name("sg-logs", create_if_missing=True)

MODEL_ID = os.environ.get("SGLANG_MODEL_ID", "openai/gpt-oss-20b")
GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU      = {"L4": "L4", "L40S": "L40S", "A100": "A100-40GB", "H100": "H100"}.get(GPU_KIND, "L4")

image = (
    Image.from_registry("docker.io/lmsysorg/sglang:dev")
    .env({
        # HuggingFace downloads
        "HF_HOME": "/hf",
        "TRANSFORMERS_CACHE": "/hf/transformers",
        "HF_DATASETS_CACHE": "/hf/datasets",
        # Compiler / JIT caches
        "TORCHINDUCTOR_CACHE_DIR": "/mnt/cachejjjjjjjjjjjjjjjjjjjj/inductor",
        "TRITON_CACHE_DIR": "/mnt/cachejjjj/triton",
        "CUDA_CACHE_PATH": "/mnt/cachejjjj/nv",
        "XDG_CACHE_HOME": "/mnt/cachejjjj/xdg",
        # Optional: FlashInfer cache (XDG covers it; set explicit path if you want)
        "FLASHINFER_CACHE_DIR": "/mnt/cachejjjj/flashinfer",
        # Stable CPU threads for repeatable timings
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        # Don’t surprise-JIT deepgemm unless you ask for it
        "SGL_ENABLE_JIT_DEEPGEMM": os.environ.get("SGL_ENABLE_JIT_DEEPGEMM", "0"),
    })
)

def _ensure():
    os.makedirs("/logs/sglang", exist_ok=True)
    os.makedirs("/logs/sglang/crash_dumps", exist_ok=True)
    os.makedirs("/logs/sglang/traces", exist_ok=True)

@app.function(image=image, gpu=None, volumes={"/hf": hf})
def prewarm(model_id: str = MODEL_ID) -> str:
    from huggingface_hub import snapshot_download
    tgt = f"/hf/models/{model_id.replace('/', '__')}"
    snapshot_download(repo_id=model_id, local_dir=tgt)
    return tgt

@app.function(image=image, gpu=GPU, volumes={"/hf": hf, "/mnt/cache": kc, "/logs": log}, timeout=30*60)
def build_kernels() -> str:
    """Optional one-shot prebuild to prime DeepGEMM/Triton/Inductor caches."""
    _ensure()
    log_path = f"/logs/sglang/build_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log"
    code = 0
    with open(log_path, "a", buffering=1) as f:
        f.write("Priming kernels…\n")
        # Try deepgemm compile if exposed; otherwise run a short generation to trigger JIT
        py = r"""
import os, torch
import sglang as sgl
try:
    from sglang import compile_deep_gemm
    compile_deep_gemm()
    print("compile_deep_gemm() done.")
except Exception as e:
    print("deepgemm not available or failed:", e)
eng = sgl.Engine(model_path=os.getenv("MODEL_ID", "openai/gpt-oss-20b"))
torch.cuda.synchronize()
_ = eng.generate(["Warm up the engine for JIT build."], {"max_new_tokens": 8, "temperature": 0.0})
torch.cuda.synchronize()
eng.shutdown()
"""
        proc = subprocess.Popen(["python", "-c", py], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "MODEL_ID": MODEL_ID})
        for line in proc.stdout or []:
            print(line, end=""); f.write(line)
        code = proc.wait()
    return f"{log_path} (exit={code})"

@app.function(image=image, gpu=GPU, volumes={"/hf": hf, "/mnt/cachejjjjjjjj": kc})
def inference_test(prompt: str = "Say hi in 5 words.", model_id: str = MODEL_ID, max_new_tokens: int = 64, attention_backend: str = "") -> dict:
    import os, time, torch, sglang as sgl
    if attention_backend:
        os.environ["SGLANG_ATTENTION_BACKEND"] = attention_backend
    t0 = time.time()
    eng = sgl.Engine(model_path=model_id)
    torch.cuda.synchronize(); t1 = time.time()
    out = eng.generate([prompt], {"max_new_tokens": max_new_tokens, "temperature": 0.0})
    torch.cuda.synchronize(); t2 = time.time()
    eng.shutdown()
    return {"load_s": round(t1 - t0, 3), "infer_s": round(t2 - t1, 3), "preview": out[0]["text"][:200]}

@app.function(image=image, gpu=GPU, volumes={"/hf": hf, "/mnt/cachejjjjjjjj": kc, "/logs": log}, timeout=24*60*60, max_containers=1)
@modal.web_server(port=8000)
def serve():
    """Run sglang.launch_server from the official image with persisted caches."""
    _ensure()
    args = ["python", "-m", "sglang.launch_server",
            "--model-path", MODEL_ID, "--host", "0.0.0.0", "--port", "8000",
            "--download-dir", "/hf",
            "--crash-dump-folder", "/logs/sglang/crash_dumps"]
    if os.environ.get("SGLANG_ATTENTION_BACKEND"):
        args += ["--attention-backend", os.environ["SGLANG_ATTENTION_BACKEND"]]
    if os.environ.get("SGLANG_SAMPLING_BACKEND"):
        args += ["--sampling-backend", os.environ["SGLANG_SAMPLING_BACKEND"]]
    if os.environ.get("SGLANG_API_KEY"):
        args += ["--api-key", os.environ["SGLANG_API_KEY"]]

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = f"/logs/sglang/server_{run_id}.log"
    with open(log_path, "a", buffering=1) as f:
        f.write(f"CMD: {' '.join(args)}\n")
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def _term(sig, frm):
            try: proc.terminate()
            except: pass
        signal.signal(signal.SIGTERM, _term)

        for line in proc.stdout or []:
            print(line, end="")
            try: f.write(line)
            except: pass

@app.function()
def endpoint_url() -> str:
    try:
        fn = modal.Function.from_name(app.name, "serve")
        return fn.get_web_url() or ""
    except Exception:
        return ""
