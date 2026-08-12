"""
inference.py — CatVTON + IP-Adapter + Garment Graphic Warp + Strict Bbox Composite
─────────────────────────────────────────────────────────────────────────────────────
Architecture:
  1. CatVTON        — handles garment SHAPE & FIT (oversized vs fitted)
  2. IP-Adapter     — injects garment CLIP features to preserve texture/color
  3. Graphic warp   — homography-warps original garment onto result (preserves text/logos)
  4. Strict bbox    — composites ONLY the clothing region; face+bg = 100% original pixels
"""

import os
import sys
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# CatVTON pipeline loader
# ─────────────────────────────────────────────────────────────────────────────
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
# Garment graphic warp — preserves text/logos that diffusion destroys
# ─────────────────────────────────────────────────────────────────────────────
def _warp_garment_graphic(
    result_img: np.ndarray,
    garment_path: str,
    parse_arr: np.ndarray,
    W: int,
    H: int,
) -> np.ndarray:
    """
    Warp the ORIGINAL garment image onto the generated result using a
    perspective transform estimated from the clothing bounding boxes.

    Why: diffusion models destroy text/logos (hallucination). This step
    pastes crisp original graphics back using a purely geometric operation.
    """
    import cv2

    garment = np.array(Image.open(garment_path).convert("RGB").resize((W, H)))

    # Build clothing mask from result image and parse map
    UPPER = {5, 6, 7}
    clothing_mask = np.zeros(parse_arr.shape, dtype=np.uint8)
    for lbl in UPPER:
        clothing_mask[parse_arr == lbl] = 255

    if clothing_mask.sum() == 0:
        print("  ⚠️  Garment warp: no clothing region found, skipping")
        return result_img

    # Source bbox: garment flat-lay (approximately full image minus margins)
    src_pts = np.float32([
        [int(W * 0.05), int(H * 0.05)],
        [int(W * 0.95), int(H * 0.05)],
        [int(W * 0.95), int(H * 0.90)],
        [int(W * 0.05), int(H * 0.90)],
    ])

    # Destination bbox: clothing region on the person
    ys, xs = np.where(clothing_mask > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    dst_pts = np.float32([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ])

    # Perspective transform: warp flat-lay garment to body-fitted region
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(garment, M, (W, H))

    # Build a soft blend mask: only blend where clothing is detected
    blend_mask = cv2.GaussianBlur(
        clothing_mask.astype(np.float32), (15, 15), 0
    ) / 255.0

    # Only warp where the original garment has content (not background white)
    garment_content = np.any(garment < 240, axis=2).astype(np.float32)
    garment_content_warped = cv2.warpPerspective(garment_content, M, (W, H))
    garment_content_warped = cv2.GaussianBlur(garment_content_warped, (11, 11), 0)

    # Final blend: 60% original graphic, 40% diffusion result
    # (full 100% warp would ignore lighting/shading from diffusion)
    alpha = blend_mask * garment_content_warped * 0.6
    alpha = alpha[:, :, np.newaxis]

    blended = (warped * alpha + result_img * (1 - alpha)).astype(np.uint8)
    print("  ✓ Garment graphic warped onto result")
    return blended


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
    import torch

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

        # ── Warp original garment graphic onto diffusion result ───────────────
        #print("  Warping original garment graphic (text/logo preservation)...")
        result_arr = _warp_garment_graphic(result_arr, garment_path, parse_arr, W, H)

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