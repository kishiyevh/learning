"""
Author: Huseyn Kishiyev
Sources: MIT Lectures, OpenCV docs and Claude for epipolar lines function.
----------------------
Estimates homography and essential matrix from two images using
ORB features + BFMatcher + RANSAC. Then recovers relative camera pose.

Works without any neural network models — good for testing the geometry pipeline.
Swap the feature extraction with SuperPoint+LightGlue for production.

Usage:
  python3 geometry_estimation.py --img0 left.png --img1 right.png \
      --fx 800 --fy 800 --cx 320 --cy 240

Approximate intrinsics for a typical 640x480 camera:
  fx=fy=800, cx=320, cy=240

Requires: opencv-python numpy matplotlib
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def extract_orb_matches(img0_gray, img1_gray, n_features=2000, match_ratio=0.75):
    """ORB detection + Lowe ratio test matching."""
    orb = cv2.ORB_create(nfeatures=n_features)
    kp0, des0 = orb.detectAndCompute(img0_gray, None)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)

    if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
        return [], [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des0, des1, k=2)

    good = []
    pts0, pts1 = [], []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < match_ratio * n.distance:
                good.append(m)
                pts0.append(kp0[m.queryIdx].pt)
                pts1.append(kp1[m.trainIdx].pt)

    return good, np.float32(pts0), np.float32(pts1)


def estimate_homography(pts0, pts1, reproj_threshold=3.0):
    if len(pts0) < 4:
        return None, None
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, reproj_threshold)
    return H, mask


def estimate_essential(pts0, pts1, K, threshold_px=1.0, prob=0.999):
    """
    Estimate essential matrix in normalized coordinates.
    threshold_px is in pixels; we convert to normalized coords.
    """
    if len(pts0) < 5:
        return None, None, None, None

    # Normalize
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts0_n = np.zeros_like(pts0)
    pts1_n = np.zeros_like(pts1)
    pts0_n[:, 0] = (pts0[:, 0] - cx) / fx
    pts0_n[:, 1] = (pts0[:, 1] - cy) / fy
    pts1_n[:, 0] = (pts1[:, 0] - cx) / fx
    pts1_n[:, 1] = (pts1[:, 1] - cy) / fy

    # Threshold in normalized coords (approx focal_avg normalized)
    focal_avg = (fx + fy) / 2.0
    thresh_n = threshold_px / focal_avg

    E, mask_e = cv2.findEssentialMat(
        pts0_n, pts1_n,
        focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC, prob=prob, threshold=thresh_n
    )
    if E is None or mask_e is None:
        return None, None, None, mask_e

    n_inliers, R, t, mask_pose = cv2.recoverPose(
        E, pts0_n, pts1_n, mask=mask_e
    )
    return E, R, t, mask_e


def draw_epipolar_lines(img0, img1, pts0, pts1, F, mask, n_lines=20):
    """Draw epipolar lines on both images for inlier matches."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    inlier_idx = np.where(mask.ravel() == 1)[0]
    sample = inlier_idx[:n_lines]

    canvas0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0.copy()
    canvas1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()

    colors = [tuple(int(c) for c in cv2.applyColorMap(
        np.array([[int(i * 255 / n_lines)]], dtype=np.uint8), cv2.COLORMAP_HSV)[0, 0])
              for i in range(n_lines)]

    for idx, color in zip(sample, colors):
        p0 = pts0[idx]
        p1 = pts1[idx]

        # Epipolar line in img1 from point in img0
        line1 = F @ np.array([p0[0], p0[1], 1.0])
        a, b, c = line1
        if abs(b) > 1e-6:
            x0, x1 = 0, w1
            y0 = int(-c / b)
            y1 = int(-(a * x1 + c) / b)
            cv2.line(canvas1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(canvas0, (int(p0[0]), int(p0[1])), 5, color, -1)
        cv2.circle(canvas1, (int(p1[0]), int(p1[1])), 5, color, -1)

    combined = np.hstack([canvas0, canvas1])
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img0", required=True)
    parser.add_argument("--img1", required=True)
    parser.add_argument("--fx",   type=float, default=800.0)
    parser.add_argument("--fy",   type=float, default=800.0)
    parser.add_argument("--cx",   type=float, default=320.0)
    parser.add_argument("--cy",   type=float, default=240.0)
    args = parser.parse_args()

    K = np.array([
        [args.fx,    0.0, args.cx],
        [   0.0, args.fy, args.cy],
        [   0.0,    0.0,    1.0 ],
    ])

    img0 = cv2.imread(args.img0)
    img1 = cv2.imread(args.img1)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    print("Extracting ORB features and matching...")
    matches, pts0, pts1 = extract_orb_matches(gray0, gray1)
    print(f"  Raw matches after ratio test: {len(matches)}")

    if len(pts0) < 8:
        print("Not enough matches. Try different images.")
        return

    # Homography
    H, mask_H = estimate_homography(pts0, pts1)
    if H is not None and mask_H is not None:
        n_h_inliers = mask_H.sum()
        print(f"\nHomography RANSAC inliers: {n_h_inliers}/{len(pts0)} "
              f"({100*n_h_inliers/len(pts0):.1f}%)")
        print(f"H:\n{H}")

    # Essential matrix + pose
    E, R, t, mask_E = estimate_essential(pts0, pts1, K)
    if E is not None:
        n_e_inliers = mask_E.sum()
        print(f"\nEssential matrix RANSAC inliers: {n_e_inliers}/{len(pts0)} "
              f"({100*n_e_inliers/len(pts0):.1f}%)")
        print(f"R:\n{R}")
        print(f"t (direction only, scale unknown): {t.ravel()}")

        # Compute F from E and K for epipolar line visualization
        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

        epi_img = draw_epipolar_lines(gray0, gray1, pts0, pts1, F, mask_E)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(cv2.cvtColor(epi_img, cv2.COLOR_BGR2RGB))
        ax.set_title("Epipolar Lines (from E, RANSAC inliers only)")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig("epipolar_lines.png", dpi=150)
        plt.show()
    else:
        print("Essential matrix estimation failed.")


if __name__ == "__main__":
    main()
