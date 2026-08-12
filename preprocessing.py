"""
preprocessing.py — Body parsing, DensePose, masking
All helpers run inside the Modal GPU container.
"""

import os
import numpy as np


def human_parsing(img_path: str, out_path: str, weights_dir: str) -> np.ndarray:
    """SCHP ONNX — 473×473 input, CPU provider, returns LIP 20-class parse array."""
    import onnxruntime as ort
    from PIL import Image

    SIZE  = 473
    img   = Image.open(img_path).convert("RGB")
    orig  = img.size
    arr   = np.array(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    mean  = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    std   = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    arr   = ((arr - mean) / std).transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    sess   = ort.InferenceSession(
        os.path.join(weights_dir, "parsing_lip.onnx"),
        providers=["CPUExecutionProvider"],
    )
    logits = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
    parse  = np.argmax(logits[0], axis=0).astype(np.uint8)
    parse_img = Image.fromarray(parse, mode="L").resize(orig, Image.NEAREST)
    parse_img.save(out_path)
    print("  ✓ Human parse map generated")
    return np.array(parse_img)


def make_agnostic(
    person_path: str,
    parse_arr: np.ndarray,
    mask_path: str,
    agnostic_path: str,
) -> None:
    """
    Generate clothing mask + agnostic person image.
    LIP labels: 5=upper-clothes, 6=dress, 7=coat (only these three)
    """
    import cv2
    from PIL import Image

    UPPER = {5, 6, 7}
    mask  = np.zeros(parse_arr.shape, dtype=np.uint8)
    for lbl in UPPER:
        mask[parse_arr == lbl] = 255

    # Precise dilation — small kernel, multiple iterations
    # Avoids the "eating into armpits" problem from large kernels
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=2)
    Image.fromarray(mask).save(mask_path)

    agnostic = np.array(Image.open(person_path).convert("RGB"))
    agnostic[mask > 0] = 128
    Image.fromarray(agnostic).save(agnostic_path)
    print("  ✓ Agnostic mask + image generated")


def make_cloth_mask(garment_path: str, out_path: str) -> None:
    """
    Alpha-aware cloth mask with morphological close to fill holes.
    Handles RGBA (PNG with transparency) and opaque RGB garments correctly.
    """
    import cv2
    from PIL import Image

    img = Image.open(garment_path)
    if img.mode == "RGBA":
        # Use actual alpha channel — most accurate for product shots
        alpha = np.array(img)[:, :, 3]
        mask  = (alpha > 10).astype(np.uint8) * 255
    else:
        arr  = np.array(img.convert("RGB"))
        # 3-channel AND threshold handles light-colored garments
        mask = (~np.all(arr > 235, axis=2)).astype(np.uint8) * 255

    # Morphological close fills small holes inside the garment silhouette
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    Image.fromarray(mask).save(out_path)
    print("  ✓ Cloth mask generated")


def run_densepose(
    img_path: str, out_path: str, weights_dir: str, W: int, H: int
) -> None:
    """
    Real DensePose IUV using detectron2.
    Falls back to UV-gradient map if detection fails.
    Config YAMLs are pre-cached in the volume — no runtime URL fetch.
    """
    import cv2
    from PIL import Image

    try:
        import torch
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from densepose import add_densepose_config
        from densepose.vis.extractor import DensePoseResultExtractor

        # Patch relative _BASE_ path to our volume-cached absolute path
        dp_yaml   = os.path.join(weights_dir, "densepose_rcnn_R_50_FPN_s1x.yaml")
        base_yaml = os.path.join(weights_dir, "Base-RCNN-FPN.yaml")
        with open(dp_yaml) as f:
            text = f.read().replace("../../Base-RCNN-FPN.yaml", base_yaml)
        patched = "/tmp/dp_patched.yaml"
        with open(patched, "w") as f:
            f.write(text)

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(patched)
        cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "model_final_162be9.pkl")
        cfg.MODEL.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.freeze()

        predictor = DefaultPredictor(cfg)
        img_bgr   = cv2.resize(cv2.imread(img_path), (W, H))
        instances = predictor(img_bgr)["instances"]

        iuv = np.zeros((H, W, 3), dtype=np.uint8)
        if len(instances) > 0 and instances.has("pred_densepose"):
            extractor      = DensePoseResultExtractor()
            results, boxes = extractor(instances)
            for result, box in zip(results.results, boxes):
                x, y, bw, bh = [int(v) for v in box]
                x2, y2 = min(x + bw, W), min(y + bh, H)
                rh, rw  = y2 - y, x2 - x
                if rh <= 0 or rw <= 0:
                    continue
                lbl = result.labels.cpu().numpy().astype(np.uint8)
                u   = (result.uv[0].cpu().numpy() * 255).astype(np.uint8)
                v   = (result.uv[1].cpu().numpy() * 255).astype(np.uint8)
                iuv[y:y2, x:x2, 0] = cv2.resize(lbl, (rw, rh), interpolation=cv2.INTER_NEAREST)
                iuv[y:y2, x:x2, 1] = cv2.resize(u,   (rw, rh), interpolation=cv2.INTER_LINEAR)
                iuv[y:y2, x:x2, 2] = cv2.resize(v,   (rw, rh), interpolation=cv2.INTER_LINEAR)
            print("  ✓ Real DensePose IUV generated")
        else:
            raise RuntimeError("no person detected")

        Image.fromarray(iuv).save(out_path)

    except Exception as e:
        print(f"  ⚠️  DensePose fallback ({e})")
        ys, xs = np.mgrid[0:H, 0:W]
        iuv    = np.zeros((H, W, 3), dtype=np.uint8)
        iuv[..., 0] = 1
        iuv[..., 1] = (ys / H * 255).astype(np.uint8)
        iuv[..., 2] = (xs / W * 255).astype(np.uint8)
        Image.fromarray(iuv).save(out_path)