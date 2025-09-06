# modal_app.py
import os, signal, subprocess, json, shutil
from datetime import datetime
import modal
from modal import Image, Volume, gpu

app = modal.App("sglang-dev")


# volumes: models, jit/compile caches, logs
hf  = Volume.from_name("hf-cache", create_if_missing=True)
kc  = Volume.from_name("kernels-cache", create_if_missing=True)
log = Volume.from_name("sg-logs", create_if_missing=True)
workspace = Volume.from_name("workspace", create_if_missing=True)

 

image = (
    Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands(
        # Prepare to add NVIDIA CUDA APT repo (for Nsight CLI tools)
        "apt-get update && apt-get install -y curl ca-certificates gnupg",
        "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb",
        "apt-get update",
    )
    .apt_install(
        # Common tools
        "git", "wget", "curl", "build-essential", "ccache", "gdb",
        # Rust toolchain + native build helpers for building Rust/Python extensions
        "cargo", "rustc", "pkg-config", "cmake", "ninja-build",
        # Kernel build deps
        "libnuma-dev",            # required by MSCCl++ and NUMA-aware components
        "rdma-core", "libibverbs-dev",  # optional RDMA/IB verbs support (non-fatal if unused)
        # Nsight CLI tools from CUDA 12.9 repo
        "cuda-nsight-systems-12-9", "cuda-nsight-compute-12-9",
    )
    .uv_pip_install(
        # Ensure uv is present for runtime `python3 -m uv ...`
        "uv",
        # Build backends/tools for no-build-isolation flows
        "scikit-build-core",   # backend for sgl-kernel
        "setuptools-rust",     # backend for sgl-router
        "ninja",
        "setuptools",
        "wheel",
        "numpy",               # quiets PyTorch's numpy warning during configure
        # Runtime deps (kept)
        "pybase64",
        "huggingface_hub",
    )
)


MODEL_ID = os.environ.get("SGLANG_MODEL_ID", "openai/gpt-oss-20b")
GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU      = {"L4": "L4", "L40S": "L40S", "A100": "A100-40GB", "H100": "H100"}.get(GPU_KIND, "L4")


def set_build_vars():
    """Set build-related environment variables only.

    Keep compiler caches; runtime-specific caches and HF-related envs are handled elsewhere.
    """
    # Compiler cache for C/C++/CUDA
    os.environ["CCACHE_DIR"] = "/kernels/build/ccache"
    # Avoid link/clone ops on Modal volumes; use copy-based temp in /tmp
    os.environ["CCACHE_HARDLINK"] = "0"
    os.environ["CCACHE_FILE_CLONE"] = "0"
    os.environ["CCACHE_TEMPDIR"] = "/tmp/ccache-tmp"
    os.environ["CC"] = "ccache gcc"
    os.environ["CXX"] = "ccache g++"

    os.environ["CARGO_HOME"] = "/kernels/build/cargo"
    os.environ["CARGO_TARGET_DIR"] = "/kernels/build/cargo/target"
    # Use ccache for C/C++/CUDA via CMake compiler launchers
    os.environ["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CMAKE_CUDA_COMPILER_LAUNCHER"] = "ccache"
    # Ensure cache/temp dirs exist
    for d in [
        os.environ.get("CCACHE_DIR", "/kernels/build/ccache"),
        os.environ.get("CARGO_HOME", "/kernels/build/cargo"),
        os.environ.get("CARGO_TARGET_DIR", "/kernels/build/cargo/target"),
        os.environ.get("CCACHE_TEMPDIR", "/tmp/ccache-tmp"),
    ]:
        os.makedirs(d, exist_ok=True)




@app.function(image=image, gpu=None, volumes={"/hf": hf})
def prewarm(model_id: str = MODEL_ID) -> str:
    from huggingface_hub import snapshot_download
    tgt = f"/hf/models/{model_id.replace('/', '__')}"
    snapshot_download(repo_id=model_id, local_dir=tgt)
    return tgt



@app.local_entrypoint()
def sync_workspace(src: str = "./sglang"):
    """Sync the local sglang directory into the 'workspace' Modal Volume.
    Usage: modal run app.py::sync_workspace --src ./sglang
    """
    print(f"Syncing {src} -> volume 'workspace' at /workspace/sglang ...")
    subprocess.run(["modal", "volume", "put", "workspace", src], check=True)
    print("Done. You can now run build_kernels_workspace or workspace_ls.")

@app.function(image=image, gpu=None, volumes={"/workspace": workspace})
def workspace_ls(path: str = "/workspace/sglang") -> list:
    """List files under the synced workspace to verify contents after `modal volume put workspace ./sglang`."""
    import os
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            p = os.path.join(root, name)
            try:
                sz = os.path.getsize(p)
            except Exception:
                sz = -1
            result.append({"path": p, "size": sz})
    return result


@app.function(image=image, gpu=None)
def tool_versions() -> dict:
    """Report versions of key toolchain components inside the image."""
    import subprocess, shlex
    checks = {
        "nvcc": "nvcc --version",
        "nsys": "nsys --version",
        "ncu": "ncu --version",
        "rustc": "rustc --version",
        "cargo": "cargo --version",
        "cmake": "cmake --version",
        "ninja": "ninja --version",
        "uv": "python3 -m uv --version",
    }
    out = {}
    for name, cmd in checks.items():
        try:
            proc = subprocess.run(shlex.split(cmd), check=False, capture_output=True, text=True)
            if proc.returncode == 0:
                out[name] = proc.stdout.strip() or proc.stderr.strip()
            else:
                out[name] = f"exit={proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        except Exception as e:
            out[name] = f"error: {e}"
    return out

@app.function(image=image, gpu=GPU, volumes={"/hf": hf, "/kernels": kc, "/logs": log, "/workspace": workspace}, timeout=30*60)
def build_source() -> str:
    """Prebuild using code synced into /workspace/sglang (from 'workspace' volume)."""
    # Set caches and env BEFORE installs so builds use the /kernels volume
    set_build_vars()

    log_path = f"/logs/sglang/build_ws_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log"
    # Easy fix: make CMake 4.0 accept older project policy floors (e.g., dlpack)
    os.environ["CMAKE_ARGS"] = "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    env = {**os.environ}

    def run_and_tee(cmd: list[str]) -> int:
        with open(log_path, "a", buffering=1) as f:
            f.write("$ " + " ".join(cmd) + "\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for line in proc.stdout or []:
                print(line, end=""); f.write(line)
            return proc.wait()

    # Debugging: Print environment, permissions, and filesystem info before installs
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: user / python / cwd ===' && whoami && id && pwd && python3 -V"])  
    run_and_tee(["python3", "-c", "import os,sys; print('sys.executable=', sys.executable); print('UV_CACHE_DIR=', os.environ.get('UV_CACHE_DIR')); print('PIP_CACHE_DIR=', os.environ.get('PIP_CACHE_DIR')); print('CCACHE_DIR=', os.environ.get('CCACHE_DIR')); print('CARGO_HOME=', os.environ.get('CARGO_HOME')); print('CARGO_TARGET_DIR=', os.environ.get('CARGO_TARGET_DIR')); print('XDG_CACHE_HOME=', os.environ.get('XDG_CACHE_HOME')); print('UV_LINK_MODE=', os.environ.get('UV_LINK_MODE'));"])
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: selected env ===' && env | sort | egrep '^(UV|PIP|CC|CARGO|XDG|TORCH|TRITON|CUDA|SGLANG|HF|TOKENIZERS)' || true"])  
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: filesystem types ===' && df -Th | sed -n '1p;/\\/kernels/p;/\\/workspace/p;/\\/tmp/p'"])  
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: dirs / perms ===' && ls -ld /kernels /kernels/build /kernels/build/uv /kernels/build/pip /kernels/build/ccache /kernels/build/cargo /kernels/build/cargo/target || true"])  
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: stat details ===' && for d in /kernels /kernels/build /kernels/build/*; do stat -c '%A %U:%G %n' \"$d\"; done || true"])  
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: write tests (kernels caches) ===' && set -e; for d in /kernels/build/uv /kernels/build/pip /kernels/build/ccache /kernels/build/cargo /kernels/build/cargo/target; do touch \"$d/.write_test\" && echo OK > \"$d/.write_test\" && ls -l \"$d/.write_test\"; done; true"])  
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: root uv cache dir (should be unused) ===' && ls -ld /root/.cache /root/.cache/uv || true"])  
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: uv version ===' && python3 -m uv --version"])  
    # Small uv cache write test with a tiny package
    run_and_tee(["bash", "-lc", "echo '=== DEBUG: uv cache write test (packaging==24.2) ===' && python3 -m uv pip install --reinstall --no-deps packaging==24.2"])  

    # Build/install editable packages from the synced workspace
    code = 0
    code |= run_and_tee(["python3", "-m", "uv", "pip", "install", "-e", "/workspace/sglang/python[runtime_common]"]) or 0
    code |= run_and_tee(["python3", "-m", "uv", "pip", "install", "--no-build-isolation", "-e", "/workspace/sglang/sgl-kernel"]) or 0
    code |= run_and_tee(["python3", "-m", "uv", "pip", "install", "--no-build-isolation", "-e", "/workspace/sglang/sgl-router"]) or 0

    kc.commit(); log.commit(); 
    return f"{log_path} (exit={code})"

@app.function(image=image, gpu=GPU, volumes={"/hf": hf, "/logs": log, "/workspace": workspace})
def inference_test(prompt: str = "Say hi in 5 words.", model_id: str = MODEL_ID, max_new_tokens: int = 64, attention_backend: str = "") -> dict:
    import os, time
    # Install sglang from the synced workspace
    subprocess.run(["python3", "-m", "pip", "install", "-e", "/workspace/sglang/python"], check=True)
    subprocess.run(["python3", "-m", "pip", "install", "-e", "/workspace/sglang/sgl-kernel"], check=False)
    subprocess.run(["python3", "-m", "pip", "install", "-e", "/workspace/sglang/sgl-router"], check=False)
    import sglang as sgl
    if attention_backend:
        os.environ["SGLANG_ATTENTION_BACKEND"] = attention_backend
    t0 = time.time()
    eng = sgl.Engine(model_path=model_id)
    t1 = time.time()
    out = eng.generate([prompt], {"max_new_tokens": max_new_tokens, "temperature": 0.0})
    t2 = time.time()
    eng.shutdown()
    result = {"timestamp": datetime.utcnow().isoformat() + "Z",
              "model_id": model_id,
              "prompt": prompt,
              "completion": out[0]["text"],
              "load_time_sec": t1-t0,
              "gen_time_sec": t2-t1,
              "gen_tokens": max_new_tokens,
              "backend": os.environ.get("SGLANG_ATTENTION_BACKEND", "")}
    with open(f"/logs/sglang/inference_ws_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json", "w") as f:
        json.dump(result, f)
    return result


@app.function(image=image, gpu=None, volumes={"/kernels": kc})
def test_cache() -> dict:
    """Smoke-test that compiler caches write to /kernels (prints to console only).

    Steps:
    1) Set build env (ccache + CMake launchers)
    2) Zero ccache stats; record config
    3) Build a tiny C project twice with CMake+Ninja so the 2nd build yields ccache hits
    4) Summarize stats and confirm files exist under /kernels/build/ccache
    """
    # 1) Build env
    set_build_vars()

    # Local helpers
    env = {**os.environ}

    def run(cmd: list[str]) -> tuple[int, str]:
        print("$ " + " ".join(cmd))
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out

    # 2) Inspect env and ccache config; zero stats
    cc_dir = os.environ.get("CCACHE_DIR", "")
    run(["bash", "-lc", "echo CCACHE_DIR=$CCACHE_DIR; echo CMAKE_C_COMPILER_LAUNCHER=$CMAKE_C_COMPILER_LAUNCHER; echo CMAKE_CUDA_COMPILER_LAUNCHER=$CMAKE_CUDA_COMPILER_LAUNCHER"])
    run(["bash", "-lc", "ls -ld /kernels /kernels/build /kernels/build/ccache || true"]) 
    run(["ccache", "-z"])  # zero stats
    _, cc_cfg = run(["ccache", "--show-config"])
    _, stats0 = run(["ccache", "-s"])  # before

    # 3) Tiny C project built twice
    work = "/tmp/ccache_smoke"
    try:
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)
        with open(os.path.join(work, "CMakeLists.txt"), "w") as f:
            f.write(
                "\n".join([
                    "cmake_minimum_required(VERSION 3.22)",
                    "project(ccache_smoke LANGUAGES C)",
                    "set(CMAKE_VERBOSE_MAKEFILE ON)",
                    "add_executable(hello main.c)",
                ])
            )
        with open(os.path.join(work, "main.c"), "w") as f:
            f.write('#include <stdio.h>\nint main(){ puts("hi"); return 0; }\n')

        build1 = os.path.join(work, "build1")
        build2 = os.path.join(work, "build2")
        for bdir in (build1, build2):
            os.makedirs(bdir, exist_ok=True)

        # First build (expect cache miss)
        run(["cmake", "-S", work, "-B", build1, "-G", "Ninja"])
        run(["cmake", "--build", build1, "-v"])
        _, stats1 = run(["ccache", "-s"])  # after first build

        # Second build in a fresh build dir (expect cache hits)
        run(["cmake", "-S", work, "-B", build2, "-G", "Ninja"])
        run(["cmake", "--build", build2, "-v"])
        _, stats2 = run(["ccache", "-s"])  # after second build
    finally:
        # leave the build dirs; they are temporary, but keeping them is fine
        pass

    # 4) Summarize filesystem evidence under /kernels
    cache_file_count = 0
    sample = []
    for root, _, files in os.walk(cc_dir or "/kernels/build/ccache"):
        for name in files:
            cache_file_count += 1
            if len(sample) < 10:
                sample.append(os.path.join(root, name))

    result = {
        "ccache_dir": cc_dir,
        "ccache_config": cc_cfg,
        "stats_before": stats0,
        "stats_after_first": stats1,
        "stats_after_second": stats2,
        "cache_file_count": cache_file_count,
        "cache_file_sample": sample,
    }
    return result
