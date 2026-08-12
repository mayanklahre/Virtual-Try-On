"""CatVTON inference with a clothing-only composite for the output."""

import os
from functools import lru_cache
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# CatVTON pipeline loader
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_pipeline(weights_dir: str):
    """
    Load CatVTON using its native pipeline class.
    (IP-Adapter and XFormers removed to prevent attention processor collisions)
    """
    import torch
    import sys
    
    # Point Python to the cloned CatVTON repository
    sys.path.insert(0, "/root/CatVTON")
    from model.pipeline import CatVTONPipeline

    print("  Loading native CatVTON pipeline...")
    pipe = CatVTONPipeline(
        base_ckpt=f"{weights_dir}/sd_inpaint",
        attn_ckpt=f"{weights_dir}/catvton",
        attn_ckpt_version="mix",
        weight_dtype=torch.float16,
        use_tf32=True,
        device='cuda'
    )
    print("  ✓ CatVTON pipeline loaded")
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Strict bbox composite — face & background = 100% original, zero diffusion
# ─────────────────────────────────────────────────────────────────────────────
def _strict_bbox_composite(
    result_arr: np.ndarray,
    original_arr: np.ndarray,
    parse_arr: np.ndarray,
) -> np.ndarray:
    """
    The core insight: diffusion should ONLY touch the clothing region.
    Everything else — face, hair, skin, background — paste the original.

    Steps:
      1. Build clothing-only mask from parse labels
      2. Erode mask slightly (keeps 2-3px buffer from skin)
      3. Feather clothing boundary (smooth edge at fabric-skin transition)
      4. Hard-freeze face + hair region (no feathering — 100% original)
    """
    import cv2

    # ── Clothing zone: what diffusion generated ───────────────────────────────
    UPPER = {5, 6, 7}
    cloth_mask = np.zeros(parse_arr.shape, dtype=np.uint8)
    for lbl in UPPER:
        cloth_mask[parse_arr == lbl] = 255
    cloth_mask = cv2.dilate(cloth_mask, np.ones((8, 8), np.uint8), iterations=2)

    # Feather the clothing boundary (smooth transition at fabric-skin edge)
    cloth_soft = cv2.GaussianBlur(cloth_mask.astype(np.float32), (21, 21), 0) / 255.0
    cloth_soft = cloth_soft[:, :, np.newaxis]

    # Composite: generated inside clothing, original outside
    composite = (result_arr * cloth_soft + original_arr * (1 - cloth_soft)).astype(np.uint8)

    # ── Hard-freeze face + hair (LIP labels 2=hair, 13=face) ─────────────────
    # These are NEVER touched by diffusion — paste original pixels directly
    FACE = {2, 13}
    face_mask = np.zeros(parse_arr.shape, dtype=np.uint8)
    for lbl in FACE:
        face_mask[parse_arr == lbl] = 255
    # Erode slightly to avoid pasting hair over collar boundary
    face_mask = cv2.erode(face_mask, np.ones((3, 3), np.uint8), iterations=1)
    # Minimal feather (3px) — enough to avoid hard edge but face stays sharp
    face_soft = cv2.GaussianBlur(face_mask.astype(np.float32), (7, 7), 0)[:, :, np.newaxis] / 255.0

    composite = (original_arr * face_soft + composite * (1 - face_soft)).astype(np.uint8)

    # ── Hard-freeze background (label 0=background) ───────────────────────────
    bg_mask = (parse_arr == 0).astype(np.uint8) * 255
    bg_mask  = cv2.erode(bg_mask, np.ones((5, 5), np.uint8), iterations=1)
    bg_soft  = cv2.GaussianBlur(bg_mask.astype(np.float32), (11, 11), 0)[:, :, np.newaxis] / 255.0
    composite = (original_arr * bg_soft + composite * (1 - bg_soft)).astype(np.uint8)

    print("  ✓ Strict bbox composite: face frozen, background preserved")
    return composite


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point called from app.py
# ─────────────────────────────────────────────────────────────────────────────
def run_catvton(
    person_path: str,
    garment_path: str,
    mask_path: str,
    parse_arr: np.ndarray,
    output_path: str,
    weights_dir: str,
    W: int = 768,
    H: int = 1024,
    steps: int = 50,
    guidance: float = 2.5,
    seed: int = 42,
) -> bool:
    try:
        # ── Load models ───────────────────────────────────────────────────────
        pipe = _load_pipeline(weights_dir)

        # Ensure all images match the expected W/H dimensions
        person_img  = Image.open(person_path).convert("RGB").resize((W, H))
        garment_img = Image.open(garment_path).convert("RGB").resize((W, H))
        mask_img    = Image.open(mask_path).convert("L").resize((W, H))

        print("  Running CatVTON diffusion...")

        # The native CatVTON pipeline takes the condition image directly
        result_img = pipe(
            person_img,
            garment_img,
            mask_img,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )[0]

        result_arr   = np.array(result_img)
        original_arr = np.array(person_img)

        # ── Strict bbox composite ─────────────────────────────────────────────
        print("  Applying strict bbox composite (face + bg freeze)...")
        result_arr = _strict_bbox_composite(result_arr, original_arr, parse_arr)

        Image.fromarray(result_arr).save(output_path)
        print(f"  ✓ Result saved → {output_path}")
        return True

    except Exception as e:
        import traceback
        print(f"❌ CatVTON inference failed:\n{traceback.format_exc()}")
        return False
