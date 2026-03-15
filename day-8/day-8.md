# Camera Models. Pinhole, Distortion, and Calibration

### Pinhole camera model

The pinhole model is the standard approximation. A 3D point `P = [X, Y, Z]` in camera coordinates projects to pixel `p = [u, v]` via:

$$u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$

In homogeneous form:

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$

Where `K` is the intrinsic matrix:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

`f_x`, `f_y` are the focal lengths in pixels. `c_x`, `c_y` is the principal point (ideally image center). For a standard camera with square pixels, `f_x ≈ f_y`, but lens manufacturing variation means they can differ slightly.

The parameter `lambda` is just `Z` (depth), the scale factor that homogeneous coordinates introduce. Dividing through gives back the 2D coordinates.

### Lens distortion

Real lenses distort the image. The two main types:

**Radial distortion** — the image is magnified differently at different distances from the center. Barrel distortion (k1 < 0) makes lines bulge outward. Pincushion distortion (k1 > 0) makes them pinch inward.

The corrected point from a distorted observed point `(x_d, y_d)` (normalized coordinates):

$$x_u = x_d (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

$$y_u = y_d (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

Where `r^2 = x_d^2 + y_d^2`.

**Tangential distortion** — from the lens not being perfectly parallel to the image plane:

$$\delta x = 2 p_1 x_d y_d + p_2 (r^2 + 2 x_d^2)$$
$$\delta y = p_1 (r^2 + 2 y_d^2) + 2 p_2 x_d y_d$$

OpenCV's full distortion vector is `[k1, k2, p1, p2, k3]`. For most lenses, k3 is negligible unless you have a very wide-angle lens.

### Calibration procedure

Camera calibration estimates K and the distortion coefficients from images of a known pattern (chessboard).

For each chessboard image, you know the 3D world positions of the corners and can detect the 2D pixel positions. The calibration solves for K and distortion coefficients to minimize reprojection error and the distance between detected corners and corners re-projected using the estimated model.

With OpenCV:

```python
import cv2
import numpy as np
import glob

pattern_size = (9, 6)  # inner corners
obj_p = np.zeros((9 * 6, 3), np.float32)
obj_p[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 1m square spacing by default

obj_points = []
img_points = []

for fname in glob.glob("calib_images/*.png"):
    img  = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        obj_points.append(obj_p)
        img_points.append(corners_refined)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)
print(f"K:\n{K}")
print(f"Distortion: {dist.ravel()}")
print(f"Reprojection error: {ret:.4f} px")
```

A reprojection error below 0.5 px is generally acceptable. Above 1 px usually means the chessboard was poorly lit, out of focus, or too few images were captured.

### Why calibration accuracy matters for visual navigation

Feature matching finds correspondences between image frames, but geometry estimation (essential matrix, homography) assumes you have correct normalized coordinates. You normalize by applying K inverse: `x_n = K^{-1} [u, v, 1]^T`. If K is wrong, the normalized coordinates are wrong, the essential matrix estimate is biased, and pose recovery will be off especially in translation direction estimation.

References:

- https://vnav.mit.edu/lectures.html (Lecture 2 and 3 — camera model, calibration)
- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Hartley & Zisserman "Multiple View Geometry in Computer Vision" — Chapters 6 and 7
