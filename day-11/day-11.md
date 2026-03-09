# February 26, 2026

## Geometric Vision Homography, Essential Matrix, and Fundamental Matrix

### Homography

A homography H is a 3x3 matrix that relates corresponding points in two images when one of these conditions holds,

1. The scene is planar (all points lie on a flat surface)
2. The camera only rotates (no translation)

The relationship in homogeneous coordinates:

$$\lambda \mathbf{p}' = \mathbf{H} \mathbf{p}$$

where `p = [u, v, 1]^T` and `p' = [u', v', 1]^T` are corresponding pixel coordinates.

H has 8 degrees of freedom (it is defined up to scale, so 9 elements - 1 = 8). To estimate H you need at least 4 point correspondences. DLT (Direct Linear Transform) sets up a system of linear equations from the correspondences and solves with SVD.

Homographies are used in image stitching, planar AR tracking, and when recovering rotation-only camera motion. For a drone hovering and rotating (no translation), the entire optical flow can be described by a homography.

### Essential matrix

The essential matrix E relates corresponding normalized image coordinates between two calibrated cameras:

$$\mathbf{x}'^T \mathbf{E} \mathbf{x} = 0$$

where `x = K^{-1} p` (normalized coordinates). E encodes the relative rotation R and translation t between the two cameras:

$$\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$$

where `[t]_x` is the skew-symmetric matrix of t. E has 5 degrees of freedom (R has 3, t has 3 but scale is not recoverable, minus 1). The minimum number of correspondences to estimate E is 5 (five-point algorithm), or 8 with the simpler 8-point algorithm (which treats E as having 8 DOF and then enforces the singular value constraint).

From E you recover (R, t) up to scale, we know the direction of translation but not the magnitude without additional information (e.g., known object size or GPS).

### Fundamental matrix

The fundamental matrix F relates raw pixel coordinates:

$$\mathbf{p}'^T \mathbf{F} \mathbf{p} = 0$$

The relationship to E: `F = K'^{-T} E K^{-1}`. F is used when we do not have intrinsic calibration (or when we want to avoid relying on it). It has 7 DOF.

The epipolar constraint is what both E and F encode: a point p in image 1 constrains the corresponding point in image 2 to lie on a line (the epipolar line). This is what makes matching efficient, we only search along a 1D line instead of the full 2D image.

### Recovering pose from E

Given E, the decomposition into (R, t) has 4 solutions. Only one of them has all scene points in front of both cameras (positive depth). So you check each solution by triangulating one point and verifying it has positive Z in both camera frames.

OpenCV implements this:

```python
# Points are in normalized coordinates (after dividing by K)
E, mask = cv2.findEssentialMat(pts1_n, pts2_n, method=cv2.RANSAC, prob=0.999, threshold=1e-3)
n_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1_n, pts2_n, mask=mask)
```

### RANSAC for robustness

Matched features always contain outliers — wrong matches that pass the descriptor distance threshold. RANSAC (Random Sample Consensus) handles this:

1. Randomly sample the minimum number of correspondences (4 for H, 5 for E)
2. Fit the model (estimate H or E from the sample)
3. Count inliers — correspondences that satisfy the epipolar/homography constraint within a threshold
4. Repeat N times, keep the model with the most inliers
5. Refit on all inliers

Number of iterations needed: `N = log(1 - p) / log(1 - w^k)` where p is desired probability of success (0.99), w is fraction of inliers, k is sample size. With 50% inliers and k=5: `N ≈ 145 iterations`.

LightGlue's output is already much cleaner than brute-force NN matching, so RANSAC sees 80-90% inlier rates on good image pairs, which makes the geometry estimation very robust.

The code `geometry_estimation.py` takes two images, runs OpenCV feature matching (ORB for simplicity, no neural network dependency), and estimates both H and E with RANSAC.

References:

- https://vnav.mit.edu/lectures.html (Lecture 4 — Geometric Vision)
- Hartley & Zisserman "Multiple View Geometry in Computer Vision" — Chapters 9-11
- https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
