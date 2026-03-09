"""
Author: Huseyn Kishiyev
Sources: Claude for setting up the synthetic env
Demonstrates the pinhole camera model and radial distortion.

Part 1: Projects 3D points onto the image plane using K matrix.
Part 2: Applies and removes radial distortion, shows the warp.
Part 3: Generates a synthetic chessboard and shows distortion effect.

No actual camera required. 

Requires: numpy, matplotlib, opencv-python
  pip install opencv-python matplotlib numpy
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Typical webcam intrinsics (approximate)
K = np.array([
    [800.0,   0.0, 320.0],
    [  0.0, 800.0, 240.0],
    [  0.0,   0.0,   1.0],
])
IMAGE_W, IMAGE_H = 640, 480

# Barrel distortion coefficients (k1 < 0 => barrel)
DIST = np.array([-0.3, 0.1, 0.0, 0.0, 0.0])  # [k1, k2, p1, p2, k3]


def project_points(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project Nx3 array of 3D points (in camera frame) to 2D pixels.
    No distortion. Returns Nx2 pixel coordinates.
    """
    assert points_3d.shape[1] == 3
    # Normalize by depth
    x_n = points_3d[:, 0] / points_3d[:, 2]
    y_n = points_3d[:, 1] / points_3d[:, 2]
    # Apply intrinsics
    u = K[0, 0] * x_n + K[0, 2]
    v = K[1, 1] * y_n + K[1, 2]
    return np.stack([u, v], axis=1)


def apply_radial_distortion(x_n: np.ndarray, y_n: np.ndarray,
                             k1: float, k2: float, k3: float = 0.0):
    """
    Apply radial distortion to normalized image coordinates.
    Returns distorted normalized coordinates.
    """
    r2 = x_n**2 + y_n**2
    factor = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    return x_n * factor, y_n * factor


def demo_projection():
    # 3D points forming a simple box in front of the camera
    box_3d = np.array([
        [-1, -1, 5],
        [ 1, -1, 5],
        [ 1,  1, 5],
        [-1,  1, 5],
        [-1, -1, 8],
        [ 1, -1, 8],
        [ 1,  1, 8],
        [-1,  1, 8],
    ], dtype=float)

    pts_2d = project_points(box_3d, K)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, IMAGE_W)
    ax.set_ylim(IMAGE_H, 0)
    ax.set_facecolor("#111")
    ax.scatter(pts_2d[:4, 0], pts_2d[:4, 1], c="cyan",   s=80, zorder=5, label="Near face (Z=5)")
    ax.scatter(pts_2d[4:, 0], pts_2d[4:, 1], c="orange", s=80, zorder=5, label="Far face (Z=8)")
    # Draw edges
    for start, end in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot([pts_2d[start,0], pts_2d[end,0]],
                [pts_2d[start,1], pts_2d[end,1]], "w-", linewidth=0.8, alpha=0.7)
    ax.set_title("Pinhole Projection — 3D Box to 2D Image Plane")
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("projection_box.png", dpi=150)
    plt.show()


def demo_distortion():
    """
    Draw a grid in the undistorted normalized coordinate space,
    then show what the same grid looks like after barrel distortion.
    """
    # Grid in normalized coords (roughly covers the image)
    lin = np.linspace(-0.4, 0.4, 20)
    gx, gy = np.meshgrid(lin, lin)
    gx = gx.ravel()
    gy = gy.ravel()

    # Apply distortion
    gx_d, gy_d = apply_radial_distortion(gx, gy, k1=-0.3, k2=0.1)

    # Convert to pixels for plotting
    def to_px(xn, yn):
        u = K[0, 0] * xn + K[0, 2]
        v = K[1, 1] * yn + K[1, 2]
        return u, v

    u_undist, v_undist = to_px(gx, gy)
    u_dist,   v_dist   = to_px(gx_d, gy_d)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ux, vy, title in [
        (axes[0], u_undist, v_undist, "Undistorted (ideal pinhole)"),
        (axes[1], u_dist,   v_dist,   "Barrel distorted (k1=-0.3)"),
    ]:
        ax.scatter(ux, vy, s=5, color="royalblue")
        ax.set_xlim(0, IMAGE_W)
        ax.set_ylim(IMAGE_H, 0)
        ax.set_title(title)
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Radial Distortion Effect on Point Grid", fontsize=13)
    plt.tight_layout()
    plt.savefig("distortion_grid.png", dpi=150)
    plt.show()


def demo_undistort_image():
    """
    Synthesize a chessboard image, apply distortion using OpenCV,
    then undistort it to compare.
    """
    # Render a synthetic chessboard
    board = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)
    sq = 40  # square size in pixels
    for r in range(IMAGE_H // sq):
        for c in range(IMAGE_W // sq):
            if (r + c) % 2 == 0:
                board[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 255
    board_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

    # Apply distortion using OpenCV remap
    map1, map2 = cv2.initUndistortRectifyMap(
        K, DIST, None, K, (IMAGE_W, IMAGE_H), cv2.CV_32FC1
    )
    # Invert: distort the image (use the distort map in reverse)
    distorted = cv2.remap(board_bgr, map1, map2, cv2.INTER_LINEAR)
    undistorted = cv2.undistort(distorted, K, DIST)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, img, title in [
        (axes[0], board_bgr,    "Original chessboard"),
        (axes[1], distorted,    "After distortion applied"),
        (axes[2], undistorted,  "After undistortion"),
    ]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle("Barrel Distortion (k1=-0.3) and Correction", fontsize=13)
    plt.tight_layout()
    plt.savefig("chessboard_distortion.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    demo_projection()
    demo_distortion()
    demo_undistort_image()
