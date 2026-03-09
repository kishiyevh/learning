"""
Author: Huseyn Kishiyev
Sources: ONNX Docs & ONNX Repo, Claude for helping to write a test script.
---------------------
Full SuperPoint + LightGlue ONNX inference pipeline.
Runs both models, visualizes matched keypoints between two images.

Models needed (download from LightGlue-ONNX releases):
  https://github.com/fabio-sim/LightGlue-ONNX/releases
    - superpoint.onnx
    - lightglue.onnx

Usage:
  python3 lightglue_pipeline.py --img0 frame_a.png --img1 frame_b.png \
      --superpoint superpoint.onnx --lightglue lightglue.onnx [--gpu]

System: Ubuntu 22.04, GTX 1650 Ti 4GB
  ORT CUDA provider fits within VRAM budget for 640x480 images.

Requirements:
  pip install onnxruntime-gpu numpy opencv-python matplotlib
"""

import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def load_gray(path: str, w: int = 640, h: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ort_input (1,1,H,W) float32, original BGR)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.resize(gray, (w, h)).astype(np.float32) / 255.0
    return gray_r[np.newaxis, np.newaxis], img


def run_superpoint(session, img_input: np.ndarray) -> dict:
    input_name   = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outs = session.run(output_names, {input_name: img_input})
    return dict(zip(output_names, outs))


def run_lightglue(session, kpts0, kpts1, desc0, desc1) -> dict:
    """
    LightGlue ONNX interface (fabio-sim export).
    Inputs:  kpts0 (1,N,2), kpts1 (1,M,2), desc0 (1,N,256), desc1 (1,M,256)
    Outputs: matches0 (N,), matches1 (M,), mscores0 (N,), mscores1 (M,)
    """
    output_names = [o.name for o in session.get_outputs()]
    inputs = {
        session.get_inputs()[0].name: kpts0,
        session.get_inputs()[1].name: kpts1,
        session.get_inputs()[2].name: desc0,
        session.get_inputs()[3].name: desc1,
    }
    outs = session.run(output_names, inputs)
    return dict(zip(output_names, outs))


def draw_matches(img0_bgr, img1_bgr,
                 kpts0, kpts1,
                 matches0, mscores0,
                 orig_scale0, orig_scale1,
                 min_score=0.1, top_k=200):
    """Draw matched keypoints side by side."""
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    canvas_h = max(h0, h1)
    canvas   = np.zeros((canvas_h, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0]      = img0_bgr
    canvas[:h1, w0:w0+w1] = img1_bgr

    # Filter by score and validity (matches0[i] >= 0 means keypoint i in img0 matched)
    valid = (matches0 >= 0) & (mscores0 >= min_score)
    idx0  = np.where(valid)[0]
    idx1  = matches0[idx0]

    # Keep top-K by score
    scores_valid = mscores0[idx0]
    if len(idx0) > top_k:
        top = np.argsort(scores_valid)[-top_k:]
        idx0, idx1, scores_valid = idx0[top], idx1[top], scores_valid[top]

    # Scale coords back to original image size
    pts0 = kpts0[idx0] * orig_scale0  # (N, 2)
    pts1 = kpts1[idx1] * orig_scale1  # (N, 2)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    cmap = plt.cm.plasma
    for i, (p0, p1, s) in enumerate(zip(pts0, pts1, scores_valid)):
        color = cmap(float(s))
        ax.plot([p0[0], p1[0] + w0], [p0[1], p1[1]],
                color=color, linewidth=0.6, alpha=0.7)
        ax.plot(p0[0], p0[1], "o", color=color, markersize=3)
        ax.plot(p1[0] + w0, p1[1], "o", color=color, markersize=3)

    ax.set_title(f"SuperPoint + LightGlue — {len(idx0)} matches (min_score={min_score})")
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, ax=ax, label="Match confidence", shrink=0.7)
    plt.tight_layout()
    plt.savefig("lightglue_matches.png", dpi=150)
    plt.show()

    return len(idx0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img0",       required=True)
    parser.add_argument("--img1",       required=True)
    parser.add_argument("--superpoint", default="superpoint.onnx")
    parser.add_argument("--lightglue",  default="lightglue.onnx")
    parser.add_argument("--gpu",        action="store_true")
    parser.add_argument("--min-score",  type=float, default=0.1)
    args = parser.parse_args()

    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.gpu else ["CPUExecutionProvider"])

    sp_session  = ort.InferenceSession(args.superpoint, providers=providers)
    lg_session  = ort.InferenceSession(args.lightglue,  providers=providers)

    target_w, target_h = 640, 480

    img0_in, img0_bgr = load_gray(args.img0, target_w, target_h)
    img1_in, img1_bgr = load_gray(args.img1, target_w, target_h)

    t0 = time.perf_counter()
    sp0 = run_superpoint(sp_session, img0_in)
    sp1 = run_superpoint(sp_session, img1_in)
    t_sp = (time.perf_counter() - t0) * 1000
    print(f"SuperPoint x2: {t_sp:.1f} ms")

    # Extract outputs — adapt names to your ONNX export
    kpts0  = sp0.get("keypoints",   sp0.get("kpts",  None))[0:1]    # (1,N,2)
    kpts1  = sp1.get("keypoints",   sp1.get("kpts",  None))[0:1]
    desc0  = sp0.get("descriptors", sp0.get("desc",  None))[0:1]    # (1,N,256)
    desc1  = sp1.get("descriptors", sp1.get("desc",  None))[0:1]

    t1 = time.perf_counter()
    lg_out = run_lightglue(lg_session, kpts0, kpts1, desc0, desc1)
    t_lg = (time.perf_counter() - t1) * 1000
    print(f"LightGlue:     {t_lg:.1f} ms")
    print(f"Total:         {t_sp + t_lg:.1f} ms")

    matches0  = lg_out.get("matches0")
    mscores0  = lg_out.get("mscores0")

    # Scale from model space (640x480) back to original image size
    orig_scale0 = np.array([img0_bgr.shape[1] / target_w,
                             img0_bgr.shape[0] / target_h])
    orig_scale1 = np.array([img1_bgr.shape[1] / target_w,
                             img1_bgr.shape[0] / target_h])

    n_matches = draw_matches(
        img0_bgr, img1_bgr,
        kpts0[0], kpts1[0],
        matches0, mscores0,
        orig_scale0, orig_scale1,
        min_score=args.min_score,
    )
    print(f"Matches drawn: {n_matches}")


if __name__ == "__main__":
    main()
