# modal_app.py
import os, signal, subprocess, json
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


def set_vars():
    os.environ["HF_HOME"] = "/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/hf/transformers"
    os.environ["HF_DATASETS_CACHE"] = "/hf/datasets"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/kernels/persist/inductor"
    os.environ["TRITON_CACHE_DIR"] = "/kernels/persist/triton"
    os.environ["CUDA_CACHE_PATH"] = "/kernels/persist/nv"
    os.environ["XDG_CACHE_HOME"] = "/kernels/persist/xdg"
    os.environ["FLASHINFER_CACHE_DIR"] = "/kernels/persist/flashinfer"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = os.environ.get("SGL_ENABLE_JIT_DEEPGEMM", "0")


# Build image from base and install local sglang
image = (
    Image.from_registry("python:3.10-slim")
    .apt_install("git", "wget", "curl", "build-essential")
    .uv_pip_install("torch", "huggingface_hub")
    .add_local_dir(local_path="sglang", remote_path="/root/sglang")
)

def _ensure():
    os.makedirs("/logs/sglang", exist_ok=True)
    os.makedirs("/logs/sglang/crash_dumps", exist_ok=True)
    os.makedirs("/logs/sglang/traces", exist_ok=True)
    # Ensure kernel cache directories exist (persisted via the /kernels volume)
    os.makedirs("/kernels/persist/inductor", exist_ok=True)
    os.makedirs("/kernels/persist/triton", exist_ok=True)
    os.makedirs("/kernels/persist/nv", exist_ok=True)
    os.makedirs("/kernels/persist/xdg", exist_ok=True)
    os.makedirs("/kernels/persist/flashinfer", exist_ok=True)

@app.function(image=image, gpu=None, volumes={"/hf": hf})
def prewarm(model_id: str = MODEL_ID) -> str:
    # Install local sglang before running prewarm
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/python"], check=True)
    # Optional: install auxiliary packages if they exist (kernels/router)
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/sgl-kernel"], check=False)
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/sgl-router"], check=False)
    from huggingface_hub import snapshot_download
    tgt = f"/hf/models/{model_id.replace('/', '__')}"
    snapshot_download(repo_id=model_id, local_dir=tgt)
    return tgt

@app.function(image=image, gpu=GPU, volumes={"/hf": hf, "/kernels": kc, "/logs": log}, timeout=30*60)
def build_kernels() -> str:
    """Optional one-shot prebuild to prime DeepGEMM/Triton/Inductor caches."""
    # Install local sglang before building kernels
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/python"], check=True)
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/sgl-kernel"], check=False)
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/sgl-router"], check=False)
    
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

@app.function(image=image, gpu=GPU, volumes={"/hf": hf, "/kernels": kc, "/logs": log})
def inference_test(prompt: str = "Say hi in 5 words.", model_id: str = MODEL_ID, max_new_tokens: int = 64, attention_backend: str = "") -> dict:
    import os, time, torch
    # Install local sglang before running inference
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/python"], check=True)
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/sgl-kernel"], check=False)
    subprocess.run(["uv", "pip", "install", "-e", "/root/sglang/sgl-router"], check=False)
    import sglang as sgl
    
    _ensure()
    if attention_backend:
        os.environ["SGLANG_ATTENTION_BACKEND"] = attention_backend
    t0 = time.time()
    eng = sgl.Engine(model_path=model_id)
    torch.cuda.synchronize(); t1 = time.time()
    out = eng.generate([prompt], {"max_new_tokens": max_new_tokens, "temperature": 0.0})
    torch.cuda.synchronize(); t2 = time.time()
    eng.shutdown()
    result = {"timestamp": datetime.utcnow().isoformat() + "Z",
              "model_id": model_id,
              "prompt": prompt,
              "completion": out[0]["text"],
              "load_time_sec": t1-t0,
              "gen_time_sec": t2-t1,
              "gen_tokens": max_new_tokens,
              "backend": os.environ.get("SGLANG_ATTENTION_BACKEND", "")}
    with open(f"/logs/sglang/inference_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json", "w") as f:
        json.dump(result, f)
    return result

@app.function(image=image, gpu=None, volumes={"/kernels": kc})
def test_vol(filename: str = "hello.txt", content: str = "hello world") -> dict:
    """Write a file to the mounted `/kernels` volume and persist it.

    Returns basic verification metadata.
    """
    path = f"/kernels/{filename}"
    existed_before = os.path.exists(path)
    with open(path, "w") as f:
        f.write(content)
    # Persist changes to the volume before exit
    kc.commit()
    existed_after = os.path.exists(path)
    with open(path, "r") as f:
        new_content = f.read()
    return {
        "path": path,
        "existed_before": existed_before,
        "existed_after": existed_after,
        "new_content": new_content,
    }

@app.function(image=image, gpu=None, volumes={"/kernels": kc, "/logs": log})
def volume_probe(marker: str = "hello", path: str = "/kernels/.probe") -> dict:
    """Write/read a marker under /kernels to verify volume mount and persistence.

    Returns metadata about previous existence, contents, and a listing of /kernels.
    """
    _ensure()
    existed_before = os.path.exists(path)
    prev_content = None
    if existed_before:
        try:
            with open(path, "r") as f:
                prev_content = f.read()
        except Exception as e:
            prev_content = f"<read_error: {e}>"

    ts = datetime.utcnow().isoformat() + "Z"
    with open(path, "w") as f:
        f.write(f"{ts} {marker}")

    existed_after = os.path.exists(path)
    with open(path, "r") as f:
        new_content = f.read()

    try:
        listing = os.listdir("/kernels")
    except Exception as e:
        listing = [f"<list_error: {e}>"]

    result = {
        "timestamp": ts,
        "path": path,
        "existed_before": existed_before,
        "prev_content": prev_content,
        "existed_after": existed_after,
        "new_content": new_content,
        "kernels_listing": listing,
    }
    with open(f"/logs/sglang/volume_probe_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json", "w") as f:
        json.dump(result, f)
    # Ensure changes are persisted to the mounted volumes before exit
    kc.commit()
    log.commit()
    return result