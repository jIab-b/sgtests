# modal_app.py
import os, signal, subprocess, json, shutil
from datetime import datetime
from typing import Optional, List
import modal
from modal import Image, Volume, gpu

app = modal.App("sglang-dev")


# volumes: models, jit/compile caches, logs
hf  = Volume.from_name("hf-cache", create_if_missing=True)
kc  = Volume.from_name("kernels-cache", create_if_missing=True)
log = Volume.from_name("sg-logs", create_if_missing=True)
workspace = Volume.from_name("workspace", create_if_missing=True)

 

image = (
    Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
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
        # Nsight CLI tools matching CUDA 12.8
        "cuda-nsight-systems-12-8", "cuda-nsight-compute-12-8",
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
    # Avoid hardlink/clone on Modal volumes; use supported boolean flags
    # Newer ccache rejects numeric booleans like "0"/"1".
    os.environ["CCACHE_NOHARDLINK"] = "true"
    os.environ["CCACHE_FILE_CLONE"] = "false"
    os.environ["CCACHE_TEMPDIR"] = "/tmp/ccache-tmp"
    # Let CMake use ccache via compiler launchers; keep CC/CXX as real compilers
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"

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



# --- helpers: put near the top of app.py (after imports) ---

def _detect_sm() -> str:
    """
    Best-effort: read compute capability from nvidia-smi.
    Fallback to 8.9 (L4) if not available.
    """
    import subprocess, re
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "nvidia-smi -q -d COMPUTE || true"],
            text=True, stderr=subprocess.STDOUT
        )
        m = re.search(r"Compute\s+Capability\s*:\s*([0-9]+)\.([0-9]+)", out)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    except Exception:
        pass
    return "8.9"


def _arch_lists(sm_csv: str | None) -> tuple[str, str]:
    """
    sm_csv: semicolon-joined list like '7.5;8.0;8.6;8.9;9.0'
    Returns (TORCH_CUDA_ARCH_LIST, FLASHINFER_CUDA_ARCH_LIST)
    """
    sm_csv = sm_csv or _detect_sm()
    items = [x.strip() for x in sm_csv.split(";") if x.strip()]
    torch_list = ";".join(items)
    flashinfer_list = ";".join(f"{int(float(x)*10)}" for x in items)  # 8.9 -> 89
    return torch_list, flashinfer_list



# --- UPDATED: build_source() ---

@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf": hf, "/kernels": kc, "/logs": log, "/workspace": workspace},
    timeout=30 * 60,
)
def build_source(
    sm_targets: str = "",     # e.g. "7.5;8.0;8.6;8.9;9.0" (empty => auto-detect single SM)
    multi_sm: bool = False    # if True and sm_targets empty, use a multi-SM default
) -> str:
    """
    Build sglang runtime + sgl-kernel + flashinfer-python into a persistent /kernels mount.
    uv's cache is redirected to /tmp/uvcache to avoid volume permission issues.
    """
    import os, shlex, subprocess
    from datetime import datetime

    # --- decide SMs ---
    if not sm_targets:
        sm_targets = "7.5;8.0;8.6;8.9;9.0" if multi_sm else _detect_sm()
    TORCH_CUDA_ARCH_LIST, FLASHINFER_CUDA_ARCH_LIST = _arch_lists(sm_targets)

    # --- persistent + temp caches ---
    os.makedirs("/kernels/torch_extensions", exist_ok=True)
    os.makedirs("/kernels/wheels", exist_ok=True)
    os.makedirs("/tmp/uvcache", exist_ok=True)
    os.makedirs("/tmp/pipcache", exist_ok=True)
    os.makedirs("/tmp/xdgcache", exist_ok=True)

    # try to avoid sticky perms from host
    try:
        subprocess.run(["bash", "-lc", "chmod -R 777 /kernels || true"], check=False)
    except Exception:
        pass

    # --- environment for builds ---
    env = {**os.environ}
    env.update({
        # make nvcc & CUDA headers/libs discoverable
        "CUDA_HOME": "/usr/local/cuda/targets/x86_64-linux",
        "CUDA_TOOLKIT_ROOT_DIR": "/usr/local/cuda/targets/x86_64-linux",
        "CUDA_NVCC_EXECUTABLE": "/usr/local/cuda/bin/nvcc",
        "PATH": f"/usr/local/cuda/bin:{env.get('PATH','')}",

        # where compiled extensions (.so/.cubin) go (PERSISTENT)
        "TORCH_EXTENSIONS_DIR": "/kernels/torch_extensions",
        "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST,
        "FLASHINFER_CUDA_ARCH_LIST": FLASHINFER_CUDA_ARCH_LIST,

        # uv/pip caches (TEMP; local FS to avoid 'Operation not permitted')
        "UV_CACHE_DIR": "/tmp/uvcache",
        "PIP_CACHE_DIR": "/tmp/pipcache",
        "XDG_CACHE_HOME": "/tmp/xdgcache",

        # build speed/compat and verbosity (cap parallelism to reduce OOM risk)
        "MAX_JOBS": "2",
        "CMAKE_BUILD_PARALLEL_LEVEL": "2",
        "USE_NINJA": "1",
        "VERBOSE": "1",
        "NINJA_STATUS": "[%f/%t @%r | %es]",
        "CMAKE_ARGS": (
            f"-DCMAKE_POLICY_VERSION_MINIMUM=3.5 "
            f"-DNO_CUDA_VERSION_CHECK=ON -DIGNORE_CUDA_VERSION_CHECK=ON "
            f"-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc "
            f"-DCMAKE_CUDA_ARCHITECTURES={FLASHINFER_CUDA_ARCH_LIST} "
            f"-DCMAKE_VERBOSE_MAKEFILE=ON "
            f"-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE "
            f"-DCMAKE_CUDA_FLAGS=\"-Xptxas -v -lineinfo\" "
            f"-Wno-dev"
        ),
    })

    # legacy symlinks some scripts expect
    subprocess.run(
        ["bash", "-lc",
         "set -e;"
         "if [ ! -e /usr/local/cuda/include ]; then ln -s targets/x86_64-linux/include /usr/local/cuda/include; fi; "
         "if [ ! -e /usr/local/cuda/lib64 ]; then ln -s targets/x86_64-linux/lib /usr/local/cuda/lib64; fi; "
        ],
        env=env, check=False
    )

    # --- logging ---
    log_path = f"/logs/sglang/build_ws_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log"
    def tee(cmd: list[str]) -> int:
        with open(log_path, "a", buffering=1) as f:
            f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end=""); f.write(line)
            return proc.wait()

    # --- diagnostics ---
    tee(["bash", "-lc", "echo '=== DEBUG: build env ==='; env | egrep '^(CUDA|TORCH|FLASHINFER|UV|PIP|XDG|CMAKE|PATH)=' | sort"])
    tee(["bash", "-lc", f"echo '=== DEBUG: archs ==='; echo TORCH_CUDA_ARCH_LIST={shlex.quote(TORCH_CUDA_ARCH_LIST)}; echo FLASHINFER_CUDA_ARCH_LIST={shlex.quote(FLASHINFER_CUDA_ARCH_LIST)}; echo CMAKE_CUDA_ARCHITECTURES={shlex.quote(FLASHINFER_CUDA_ARCH_LIST)}"])
    tee(["bash", "-lc", "echo '=== DEBUG: volumes ==='; df -Th | sed -n '1p;/\\/kernels/p;/\\/workspace/p;/\\/tmp/p'"])
    tee(["bash", "-lc", "echo '=== DEBUG: torch_extensions ==='; ls -lah /kernels/torch_extensions || true"])

    code = 0

    # --- 1) sglang runtime (editable) ---
    # keep uv cache on /tmp; compile-time artifacts land in TORCH_EXTENSIONS_DIR
    code |= tee([
        "python3", "-m", "uv", "pip", "install", "-v",
        "--cache-dir", "/tmp/uvcache",
        "-e", "/workspace/sglang/python[runtime_common]",
    ]) or 0

    # --- 2) sgl-kernel (force source build) ---
    code |= tee([
        "python3", "-m", "uv", "pip", "install", "-v",
        "--cache-dir", "/tmp/uvcache",
        "--no-build-isolation",
        "--no-binary", "sgl-kernel",
        "-e", "/workspace/sglang/sgl-kernel",
    ]) or 0

    # --- 3) flashinfer-python (force source build, pinned) ---
    code |= tee([
        "python3", "-m", "uv", "pip", "install", "-v",
        "--cache-dir", "/tmp/uvcache",
        "--no-build-isolation",
        "--no-binary", "flashinfer-python",
        "flashinfer-python==0.3.0",
    ]) or 0

    # (wheel-building step removed on request)

    # persist volumes
    try:
        kc.commit()
    except Exception:
        pass
    try:
        log.commit()
    except Exception:
        pass

    return f"{log_path} (exit={code})"
