"""
app.py — Virtual Try-On  (CatVTON + IP-Adapter + Strict Inpaint)
─────────────────────────────────────────────────────────────────
Commands:
  Download weights (run once): python3 -m modal run app.py::app.download_weights
  Dev server:                  python3 -m modal serve app.py
  Production deploy:           python3 -m modal deploy app.py
  CLI test:                    python3 -m modal run app.py
"""

import modal, os, shutil, json

volume = modal.Volume.from_name("vto-model-weights")

# ─────────────────────────────────────────────────────────────────────────────
# Container image
# Pin numpy first, then build everything else on top
# ─────────────────────────────────────────────────────────────────────────────
vto_image = (
    # Use CUDA 12.1 + cuDNN 8 base — required for detectron2 to compile
    # and compatible with torch 2.1.2 (cuDNN 8.x)
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0",
        "wget", "gcc", "g++", "build-essential",
        "python3-pip",
    )
    # Layer 1: pin numpy + torch FIRST before anything else runs
    # This prevents CatVTON requirements.txt from upgrading torch to 2.4
    .pip_install("numpy==1.26.4")
    .pip_install(
        "torch==2.1.2+cu121",
        "torchvision==0.16.2+cu121",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Layer 2: all other Python deps (pinned torch survives because it's already installed)
    .pip_install(
        "diffusers==0.27.2",
        "transformers==4.38.2",
        "accelerate==0.27.2",
        "huggingface_hub==0.23.0",
        "peft==0.9.0",
        "opencv-python",
        "einops",
        "ninja",
        "tqdm",
        "Pillow",
        "onnxruntime==1.16.3",
        "scipy",
        "scikit-image",
        "fvcore",
        "omegaconf",
        "pycocotools",
        "cloudpickle",
        "fastapi",
        "python-multipart",
        "safetensors",
        "xformers==0.0.23",
    )
    # Layer 3: compiled repos — torch is already locked so detectron2 builds cleanly
    .run_commands(
        # Clone CatVTON (we skip its requirements.txt — already installed above)
        "git clone https://github.com/Zheng-Chong/CatVTON.git /root/CatVTON",
        # Build detectron2 against the locked torch 2.1.2 + CUDA 12.1
        "pip install 'git+https://github.com/facebookresearch/detectron2.git'",
        # DensePose subproject
        "pip install 'git+https://github.com/facebookresearch/detectron2.git"
        "#subdirectory=projects/DensePose'",
        # Final re-pin: detectron2 sometimes pulls numpy 2.x as transitive dep
        "pip install 'numpy==1.26.4' --force-reinstall --no-deps",
    )
    # Layer 4: local files always last
    .add_local_file("./preprocessing.py",    remote_path="/root/preprocessing.py")
    .add_local_file("./inference.py",         remote_path="/root/inference.py")
    .add_local_file("./frontend/index.html",  remote_path="/root/frontend/index.html")
    .add_local_dir("./data",                  remote_path="/root/data")
)

app = modal.App("vto-catvton", image=vto_image)


# ─────────────────────────────────────────────────────────────────────────────
# Weight downloader — run once, saves everything to persistent volume
# python3 -m modal run app.py::app.download_weights
# ─────────────────────────────────────────────────────────────────────────────
@app.function(volumes={"/weights": volume}, timeout=7200, cpu=4)
def download_weights():
    import subprocess
    import urllib.request
    from huggingface_hub import snapshot_download, hf_hub_download

    # ── CatVTON attention checkpoints ────────────────────────────────────────
    print("⬇️  CatVTON attention weights...")
    os.makedirs("/weights/catvton", exist_ok=True)
    snapshot_download(
        repo_id="zhengchong/CatVTON",
        local_dir="/weights/catvton",
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print("  ✓ CatVTON weights")

    # ── SD Inpainting base model (CatVTON backbone) ───────────────────────────
    print("⬇️  Stable Diffusion Inpainting base model (~4 GB)...")
    os.makedirs("/weights/sd_inpaint", exist_ok=True)
    snapshot_download(
        repo_id="runwayml/stable-diffusion-inpainting",
        local_dir="/weights/sd_inpaint",
        ignore_patterns=["*.md", ".gitattributes", "*.ckpt"],
    )
    print("  ✓ SD Inpainting base")

    # ── IP-Adapter-Plus (garment feature encoder) ────────────────────────────
    print("⬇️  IP-Adapter-Plus weights + image encoder...")
    os.makedirs("/weights/ip_adapter/models", exist_ok=True)
    os.makedirs("/weights/ip_adapter/models/image_encoder", exist_ok=True)

    # Image encoder (ViT-H — high detail, essential for text/logo preservation)
    snapshot_download(
        repo_id="h94/IP-Adapter",
        allow_patterns=["models/image_encoder/**", "models/ip-adapter-plus_sd15.bin"],
        local_dir="/weights/ip_adapter",
    )
    print("  ✓ IP-Adapter-Plus")

    # ── SCHP human parsing ONNX ───────────────────────────────────────────────
    print("⬇️  SCHP human parsing ONNX...")
    os.makedirs("/weights/preprocess/humanparsing", exist_ok=True)
    for fname in ["parsing_atr.onnx", "parsing_lip.onnx"]:
        cached = hf_hub_download(
            repo_id="levihsu/OOTDiffusion",
            filename=f"checkpoints/humanparsing/{fname}",
        )
        shutil.copy(cached, f"/weights/preprocess/humanparsing/{fname}")
    print("  ✓ SCHP parsing")

    # ── DensePose R50-FPN weights + configs ───────────────────────────────────
    print("⬇️  DensePose weights + configs...")
    os.makedirs("/weights/preprocess/densepose", exist_ok=True)
    subprocess.run([
        "wget", "-q", "-O",
        "/weights/preprocess/densepose/model_final_162be9.pkl",
        "https://dl.fbaipublicfiles.com/densepose/"
        "densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
    ], check=True)
    for fname, url in {
        "Base-RCNN-FPN.yaml":
            "https://raw.githubusercontent.com/facebookresearch/detectron2/"
            "main/configs/Base-RCNN-FPN.yaml",
        "densepose_rcnn_R_50_FPN_s1x.yaml":
            "https://raw.githubusercontent.com/facebookresearch/detectron2/"
            "main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
    }.items():
        urllib.request.urlretrieve(url, f"/weights/preprocess/densepose/{fname}")
    print("  ✓ DensePose")

    volume.commit()
    print("\n✅ All weights saved to volume!")
    print("   CatVTON:      /weights/catvton")
    print("   SD Inpaint:   /weights/sd_inpaint")
    print("   IP-Adapter:   /weights/ip_adapter")
    print("   SCHP:         /weights/preprocess/humanparsing")
    print("   DensePose:    /weights/preprocess/densepose")


# ─────────────────────────────────────────────────────────────────────────────
# GPU inference function
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    gpu="A10G",
    volumes={"/weights": volume},
    timeout=600,
    scaledown_window=60,
)
def run_vto_inference(person_bytes: bytes, garment_bytes: bytes):
    import sys, io
    from PIL import Image

    sys.path.insert(0, "/root")
    from preprocessing import (
        human_parsing, make_agnostic, make_cloth_mask, run_densepose,
    )
    from inference import run_catvton

    W, H = 768, 1024
    WEIGHTS = "/weights"

    # ── Save inputs ───────────────────────────────────────────────────────────
    os.makedirs("/tmp/vto", exist_ok=True)
    person_path  = "/tmp/vto/person.jpg"
    garment_path = "/tmp/vto/garment.jpg"
    result_path  = "/tmp/vto/result.png"

    def save_rgb(data: bytes, path: str):
        Image.open(io.BytesIO(data)).convert("RGB").resize((W, H)).save(path)

    save_rgb(person_bytes,  person_path)
    save_rgb(garment_bytes, garment_path)

    # ── Step 1: Human parsing ─────────────────────────────────────────────────
    print("🧠 Step 1/4 — Human body parsing...")
    parse_out  = "/tmp/vto/parse.png"
    parse_arr  = human_parsing(
        person_path, parse_out, f"{WEIGHTS}/preprocess/humanparsing"
    )

    # ── Step 2: Generate clothing mask + agnostic image ───────────────────────
    print("✂️  Step 2/4 — Generating clothing mask...")
    mask_path      = "/tmp/vto/mask.png"
    agnostic_path  = "/tmp/vto/agnostic.jpg"
    make_agnostic(person_path, parse_arr, mask_path, agnostic_path)

    # ── Step 3: DensePose body map ────────────────────────────────────────────
    print("📐 Step 3/4 — DensePose body map...")
    densepose_path = "/tmp/vto/densepose.jpg"
    run_densepose(
        person_path, densepose_path,
        f"{WEIGHTS}/preprocess/densepose", W, H
    )

    make_cloth_mask(garment_path, "/tmp/vto/cloth_mask.png")

    # ── Step 4: CatVTON + IP-Adapter inference ────────────────────────────────
    print("🎨 Step 4/4 — CatVTON inference (50 steps)...")
    success = run_catvton(
        person_path   = person_path,
        garment_path  = garment_path,
        mask_path     = mask_path,
        parse_arr     = parse_arr,
        output_path   = result_path,
        weights_dir   = WEIGHTS,
        W=W, H=H,
        steps=50,
        guidance=2.5,
    )

    if not success or not os.path.exists(result_path):
        print("❌ Inference failed")
        return None

    with open(result_path, "rb") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# Web App
# ─────────────────────────────────────────────────────────────────────────────
@app.function(timeout=660)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import HTMLResponse, Response

    api = FastAPI(title="Virtual Try-On")

    @api.get("/", response_class=HTMLResponse)
    async def ui():
        with open("/root/frontend/index.html") as f:
            return f.read()

    @api.post("/try-on")
    async def try_on(
        person:  UploadFile = File(...),
        garment: UploadFile = File(...),
    ):
        person_bytes  = await person.read()
        garment_bytes = await garment.read()
        # .aio() = non-blocking async Modal remote call (required inside async FastAPI)
        result = await run_vto_inference.remote.aio(person_bytes, garment_bytes)
        if result is None:
            raise HTTPException(status_code=500, detail="Inference failed")
        return Response(content=result, media_type="image/png")

    return api


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("🚀 CLI test...")
    with open("data/person.jpg",  "rb") as f: person_bytes  = f.read()
    with open("data/garment.jpg", "rb") as f: garment_bytes = f.read()
    result = run_vto_inference.remote(person_bytes, garment_bytes)
    if result:
        with open("vto_result.png", "wb") as f: f.write(result)
        print("✅ Done — open vto_result.png")
    else:
        print("🛑 Failed — check server logs")