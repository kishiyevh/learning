"""
Author: Huseyn Kishiyev
Sources: Links mentioned in the .md and Claude
---------------------
Synthetic triangulation test.
Creates two cameras with known relative pose, projects 3D points,
adds pixel noise, triangulates, and reports 3D reconstruction error.

Good for validating the geometry pipeline before using real images.

Requires: numpy, matplotlib, opencv-python
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [        0,          0, 1],
    ])


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [         0, 1,         0],
        [-np.sin(a), 0, np.cos(a)],
    ])


def project_points(pts3d, K, R, t) -> np.ndarray:
    """Project Nx3 points into camera (R, t) with intrinsics K. Returns Nx2 pixels."""
    # Transform to camera frame
    pts_cam = (R @ pts3d.T + t.reshape(3, 1)).T  # Nx3
    # Filter behind camera
    in_front = pts_cam[:, 2] > 0
    pts_img = np.zeros((len(pts3d), 2)) * np.nan
    proj = (K @ pts_cam[in_front].T)  # 3xN
    pts_img[in_front] = (proj[:2] / proj[2]).T
    return pts_img, in_front


def dlt_triangulate(p0, p1, P0, P1) -> np.ndarray:
    """
    DLT triangulation for a single point correspondence.
    p0, p1: (2,) pixel coords (inhomogeneous)
    P0, P1: 3x4 projection matrices
    Returns: (4,) homogeneous 3D point (X, Y, Z, W)
    """
    A = np.array([
        p0[0] * P0[2] - P0[0],
        p0[1] * P0[2] - P0[1],
        p1[0] * P1[2] - P1[0],
        p1[1] * P1[2] - P1[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X / X[3]  # dehomogenize


def run_test(n_points=100, pixel_noise_std=0.5, baseline_m=0.3, pan_angle_deg=10.0):
    """
    Setup two cameras, generate random 3D points, project, add noise, triangulate.
    Returns mean and max 3D error.
    """
    np.random.seed(0)

    # Camera intrinsics
    K = np.array([
        [800.0,   0.0, 320.0],
        [  0.0, 800.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])
    W, H = 640, 480

    # Camera 0: at origin, looking along +Z
    R0 = np.eye(3)
    t0 = np.zeros(3)

    # Camera 1: translated right + slightly rotated (pan)
    R1 = rotation_matrix_y(-pan_angle_deg)
    t1 = np.array([baseline_m, 0.0, 0.0])
    t1_cam = -R1 @ t1  # t in camera frame: t_cam = -R * T_world

    # Build projection matrices
    P0 = K @ np.hstack([R0, t0.reshape(3, 1)])
    P1 = K @ np.hstack([R1, t1_cam.reshape(3, 1)])

    # Generate random 3D points in a volume in front of the cameras
    pts3d_gt = np.random.uniform(low=[-2, -2, 4], high=[2, 2, 10], size=(n_points, 3))

    # Project into both cameras
    pts2d_0, valid0 = project_points(pts3d_gt, K, R0, t0)
    pts2d_1, valid1 = project_points(pts3d_gt, K, R1, t1_cam)

    # Keep only points visible in both cameras and within image bounds
    visible = valid0 & valid1
    for pts, vld in [(pts2d_0, valid0), (pts2d_1, valid1)]:
        in_img = (pts[:, 0] >= 0) & (pts[:, 0] < W) & \
                 (pts[:, 1] >= 0) & (pts[:, 1] < H)
        visible &= in_img

    visible_idx = np.where(visible)[0]
    n_vis = len(visible_idx)

    # Add pixel noise
    pts2d_0_noisy = pts2d_0[visible_idx] + np.random.randn(n_vis, 2) * pixel_noise_std
    pts2d_1_noisy = pts2d_1[visible_idx] + np.random.randn(n_vis, 2) * pixel_noise_std

    # Triangulate using OpenCV (cleaner than looping DLT)
    pts_hom = cv2.triangulatePoints(
        P0.astype(np.float32),
        P1.astype(np.float32),
        pts2d_0_noisy.T.astype(np.float32),
        pts2d_1_noisy.T.astype(np.float32),
    )  # 4 x N
    pts3d_est = (pts_hom[:3] / pts_hom[3]).T  # N x 3

    # Compute errors
    gt_vis = pts3d_gt[visible_idx]
    errors = np.linalg.norm(pts3d_est - gt_vis, axis=1)

    print(f"Points visible in both views: {n_vis}/{n_points}")
    print(f"Pixel noise std: {pixel_noise_std:.2f} px")
    print(f"Baseline: {baseline_m:.2f} m, Pan: {pan_angle_deg:.1f} deg")
    print(f"Mean 3D error:   {errors.mean()*1000:.2f} mm")
    print(f"Median 3D error: {np.median(errors)*1000:.2f} mm")
    print(f"Max 3D error:    {errors.max()*1000:.2f} mm")

    # 3D plot
    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.scatter(gt_vis[:, 0],    gt_vis[:, 1],    gt_vis[:, 2],    s=10, c="green",  alpha=0.5, label="Ground truth")
    ax3d.scatter(pts3d_est[:, 0], pts3d_est[:, 1], pts3d_est[:, 2], s=10, c="red",   alpha=0.5, label="Triangulated")
    # Camera 0
    ax3d.scatter([0], [0], [0], s=100, c="blue", marker="^", label="Cam 0")
    ax3d.scatter([t1[0]], [t1[1]], [t1[2]], s=100, c="orange", marker="^", label="Cam 1")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend(fontsize=7)
    ax3d.set_title("3D Reconstruction")

    # Error histogram
    ax2 = fig.add_subplot(122)
    ax2.hist(errors * 1000, bins=30, color="steelblue", edgecolor="k", alpha=0.8)
    ax2.set_xlabel("3D error (mm)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Reconstruction error (noise={pixel_noise_std:.1f}px)")
    ax2.axvline(errors.mean() * 1000, color="r", linestyle="--", label=f"Mean {errors.mean()*1000:.1f}mm")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("triangulation_result.png", dpi=150)
    plt.show()

    return errors


if __name__ == "__main__":
    print("=== Test 1: Low noise ===")
    run_test(pixel_noise_std=0.5, baseline_m=0.3)
    print()
    print("=== Test 2: High noise ===")
    run_test(pixel_noise_std=2.0, baseline_m=0.3)
    print()
    print("=== Test 3: Narrow baseline ===")
    run_test(pixel_noise_std=0.5, baseline_m=0.05)
