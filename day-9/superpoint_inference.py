"""
Author: Huseyn Kishiyev
Sources: MIT Lectures, SuperPoint repo
-----------------------
Loads SuperPoint as an ONNX model and runs inference on an image.
Visualizes detected keypoints and their response (confidence) values.

SuperPoint ONNX model can be obtained from:
  https://github.com/fabio-sim/LightGlue-ONNX
  (look for superpoint.onnx in the releases or export it yourself)

Export yourself (requires PyTorch SuperPoint weights):
  https://github.com/cvg/LightGlue (pytorch implementation)

Usage:
  python3 superpoint_inference.py --image test.png --model superpoint.onnx

Requirements:
  pip install onnxruntime numpy opencv-python matplotlib
  For GPU: pip install onnxruntime-gpu

Note on GTX 1650 Ti (4GB VRAM):
  ONNX Runtime CUDAExecutionProvider works fine on this card.
  Typical inference time: ~8-12 ms per image (640x480).
"""

import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(path: str, target_size=(640, 480)) -> tuple[np.ndarray, np.ndarray]:
    """Load image, resize, normalize to [0,1] float32 grayscale."""
    img_bgr  = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, target_size)
    img_f32  = img_resized.astype(np.float32) / 255.0
    # Model expects (1, 1, H, W)
    img_input = img_f32[np.newaxis, np.newaxis]
    return img_input, img_bgr


def run_superpoint(session, img_input: np.ndarray,
                   keypoint_threshold: float = 0.005,
                   nms_radius: int = 4) -> dict:
    """
    Run SuperPoint ONNX inference.

    Expected ONNX input:  'image'  shape (1,1,H,W)
    Expected ONNX outputs:
        'keypoints'   (1, N, 2)  -- [x, y] in image pixels
        'scores'      (1, N)     -- confidence per keypoint
        'descriptors' (1, N, 256)

    Note: different ONNX exports may use different output names.
    Check session.get_outputs() to adapt if yours differs.
    """
    input_name  = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    t0 = time.perf_counter()
    outputs = session.run(output_names, {input_name: img_input})
    elapsed = (time.perf_counter() - t0) * 1000

    result = dict(zip(output_names, outputs))
    result["inference_ms"] = elapsed
    return result


def visualize_keypoints(img_bgr: np.ndarray, keypoints: np.ndarray,
                        scores: np.ndarray, title="SuperPoint Keypoints",
                        top_k: int = 500):
    """Draw top_k keypoints colored by score on image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if len(keypoints) > top_k:
        idx = np.argsort(scores)[-top_k:]
        keypoints = keypoints[idx]
        scores    = scores[idx]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img_rgb)

    sc = ax.scatter(keypoints[:, 0], keypoints[:, 1],
                    c=scores, cmap="plasma", s=15, linewidths=0, alpha=0.8,
                    vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Keypoint confidence")
    ax.set_title(f"{title}  —  {len(keypoints)} keypoints shown (top-{top_k})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("superpoint_keypoints.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="superpoint.onnx", help="Path to SuperPoint ONNX model")
    parser.add_argument("--threshold", type=float, default=0.005, help="Keypoint score threshold")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA (requires onnxruntime-gpu)")
    args = parser.parse_args()

    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    session   = ort.InferenceSession(args.model, providers=providers)

    active_provider = session.get_providers()[0]
    print(f"Using execution provider: {active_provider}")

    img_input, img_bgr = load_image(args.image)
    target_h, target_w = img_input.shape[2], img_input.shape[3]
    orig_h,   orig_w   = img_bgr.shape[:2]

    result = run_superpoint(session, img_input, keypoint_threshold=args.threshold)

    print(f"Inference time: {result['inference_ms']:.1f} ms")

    # Extract and unwrap batch dimension
    kpts   = result.get("keypoints",   result.get("kpts", None))
    scores = result.get("scores",      result.get("score", None))
    descs  = result.get("descriptors", result.get("desc",  None))

    if kpts is None:
        print("Could not find keypoints output. Available:", list(result.keys()))
        return

    kpts   = kpts[0]    # (N, 2)
    scores = scores[0]  # (N,)
    descs  = descs[0]   # (N, 256)

    print(f"Keypoints detected: {len(kpts)}")
    print(f"Descriptor shape:   {descs.shape}")
    print(f"Score range:        {scores.min():.4f} to {scores.max():.4f}")

    # Scale keypoints back to original image size for visualization
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h
    kpts_orig = kpts * np.array([scale_x, scale_y])

    visualize_keypoints(img_bgr, kpts_orig, scores)


if __name__ == "__main__":
    main()
