# SuperPoint and how Learned Sparse Feature Detection Works

### Sparse vs dense features

A sparse feature detector picks a limited set of salient points in the image, corners, edges, blobs and describes each with a compact vector (descriptor). You end up with maybe 300-1000 keypoints per image.

A dense method (like optical flow with RAFT, or feature maps from a CNN for metric localization) produces a feature for every pixel or every grid cell. This is more complete but computationally heavier and harder to use for geometric estimation, we don't want to run RANSAC on 100,000 correspondences.

For visual odometry and localization tasks where you need to estimate relative camera pose from two frames, sparse features + RANSAC is the practical choice. The geometry estimators (essential matrix, PnP) need relatively few correspondences to work well, and sparse matching keeps the pipeline fast.

### What SuperPoint does

SuperPoint is a self-supervised detector-descriptor network. It has two outputs from a shared encoder backbone:

1. A keypoint detector head — outputs a per-pixel probability that a keypoint exists there
2. A descriptor head — outputs a 256-dim descriptor at each pixel location

The architecture uses a VGG-style encoder that downsamples the input 8x. The detector head then has a channel for each position in the 8x8 cell (64 channels) plus a "no keypoint" channel (65 total), so it performs a 65-way softmax per 8x8 cell. This "dustbin" formulation avoids having to predict exact sub-pixel positions in the decoder, we get one keypoint per cell at most, or no keypoint (the dustbin wins).

The descriptor head outputs a 256-dim vector at each of the 8x-downsampled spatial locations, then bilinear interpolates these to the keypoint positions.

### Self-supervised training via homographic adaptation

This is the interesting part. You cannot easily label ground-truth keypoints on real images. SuperPoint solves this by:

1. Pre-training a simple "MagicPoint" detector on synthetic shapes (lines, polygons, cubes) where ground-truth corners are exact.
2. Applying random homographies to real images and using the MagicPoint detector to label corresponding point pairs across the warped views. Points that are consistently detected across many random warps become the training labels for the final SuperPoint network.

This generates large-scale pseudo ground-truth without manual annotation.

### SuperPoint is better than Harris/ORB

Harris corners detect any local gradient structure and they don't care if the corner is actually useful for matching across viewpoint changes. SuperPoint is trained with the objective of being repeatable across homographies, so the points it selects are specifically those that survive viewpoint changes well.

ORB descriptors are binary (Hamming distance matching), which is fast but less discriminative than the 256-dim float descriptors SuperPoint produces. In low-texture or repetitive-texture scenes, ORB confuses similar-looking patches. SuperPoint's learned descriptors capture more global context from the CNN's receptive field.

### What the inference pipeline looks like

```
Input image (grayscale)
    |
VGG encoder (8x downsample)
    |
    +-- Detector head --> 65-way softmax per 8x8 cell
    |       --> non-max suppression --> keypoint coordinates
    |
    +-- Descriptor head --> 256-dim per spatial location
            --> bilinear sample at keypoint locations
            --> L2 normalize --> final descriptors

Output: (N, 2) keypoint coords + (N, 256) descriptors
```

The entire thing runs as a single forward pass. On a GPU it is fast like under 10 ms for a 480x640 image on a decent card. On the GTX 1650 Ti (4GB) it fits comfortably for inference.

I wrote `superpoint_onnx_test.py` to load the ONNX export of SuperPoint and run it on a test image. The ONNX model is available from several repos including the LightGlue-ONNX one.

### Descriptor space and matching

Descriptors live in a 256-dimensional L2-normalized space (unit sphere). Matching is done by cosine similarity (equivalent to dot product when normalized). The LightGlue matcher refines these matches with an attention-based neural network, which we will look at in detail later.

References:

- DeTone, D., Malisiewicz, T., Rabinovich, A. "SuperPoint: Self-Supervised Interest Point Detection and Description" (2018) — arXiv:1712.07629
- https://github.com/magicleap/SuperPointPretrainedNetwork
- https://github.com/cvg/LightGlue
- https://vnav.mit.edu/lectures.html (Lecture 5 — Feature Detection)
