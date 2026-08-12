"""Body parsing and inpainting-mask helpers for the Modal container."""

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
    # The OOTDiffusion SCHP export expects OpenCV's BGR channel order.
    arr   = arr[:, :, ::-1]
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


def make_inpaint_mask(parse_arr: np.ndarray, mask_path: str) -> None:
    """
    Generate the clothing-region inpainting mask.
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
    print("  ✓ Inpainting mask generated")
