# Feature Descriptor Spaces and Matching Strategies

### Descriptor space geometry

A descriptor is a vector in R^d, for SuperPoint it is R^256, L2-normalized (lies on the unit sphere S^255). For ORB it is a binary vector in {0,1}^256 (actually 32 bytes). For SIFT it is R^128.

The important operation is distance computation. For float descriptors, the standard is either;

L2 distance (Euclidean): `||a - b||_2 = sqrt(sum((a_i - b_i)^2))`

Cosine distance: `1 - (a . b) / (||a|| ||b||)`

For L2-normalized descriptors (like SuperPoint), cosine similarity equals the dot product: `a . b`. And L2 distance is related: `||a - b||^2 = 2 - 2(a.b)`. So dot product and L2 distance give the same nearest-neighbor result for unit vectors, we usually pick whichever is faster.

For GPU inference, batched dot products via matrix multiplication are extremely efficient; if you have N descriptors from image 0 and M from image 1 (both stored as rows), the full pairwise similarity matrix is just:

```python
sim_matrix = desc0 @ desc1.T   # (N, M) — one matrix multiply on GPU
```

This is how brute-force NN matching works in CUDA. For N=M=500, this is a 500x500 matrix multiply, trivial for any GPU.

### Lowe's ratio test

Brute-force nearest neighbor returns many false matches, descriptors that are similar but correspond to different physical points (confusion in repetitive textures, viewpoint-induced appearance change). Lowe's ratio test filters these by requiring the nearest neighbor to be significantly closer than the second nearest:

```
d(desc_A, nn1) / d(desc_A, nn2) < ratio_threshold
```

A threshold of 0.75-0.8 is standard. The intuition: if there are two nearly equally good matches, the match is ambiguous and should be rejected. Only keep matches where the best candidate is clearly better than alternatives.

With SuperPoint + LightGlue, the ratio test is replaced by LightGlue's learned confidence — LightGlue already handles the rejection via its dustbin mechanism, so you apply a threshold on the output match scores rather than running a ratio test on raw descriptor distances.

### Mutual nearest neighbor matching

An alternative to ratio test: a match (i, j) is valid only if j is the nearest neighbor of i in desc1 AND i is the nearest neighbor of j in desc0. This bidirectional consistency check eliminates most false positives.

```python
nn_01 = np.argmax(sim_matrix, axis=1)   # for each pt in img0, best match in img1
nn_10 = np.argmax(sim_matrix, axis=0)   # for each pt in img1, best match in img0

# Mutual nearest neighbor: i->j and j->i
valid = np.array([nn_10[nn_01[i]] == i for i in range(len(nn_01))])
```

MNN matching is more restrictive than ratio test but produces cleaner matches in repetitive scenes.

### RANSAC, scoring the geometry

After getting matches (whether from ratio test, MNN, or LightGlue), you run RANSAC to find the subset consistent with a geometric model (essential matrix). The scoring function matters:

- Standard RANSAC: binary inlier/outlier based on threshold
- MSAC (M-estimator RANSAC): smooth penalty, more robust to the threshold choice
- MAGSAC (sigma-consensus): works without a fixed threshold by integrating over possible sigma values

OpenCV's `cv2.findEssentialMat` with `cv2.RANSAC` uses standard RANSAC. For more robustness, `cv2.USAC_MAGSAC` is now available in OpenCV 4.5+:

```python
E, mask = cv2.findEssentialMat(pts1_n, pts2_n, method=cv2.USAC_MAGSAC, prob=0.999)
```

MAGSAC generally gives fewer but cleaner inliers with better geometric accuracy, especially when there is a mix of outliers at different error scales (some matches are slightly wrong, some are completely wrong).

### Descriptor matching at scale and index structures

For large-scale retrieval (matching against a database of 10,000 images), brute-force matching is too slow. You use approximate nearest neighbor search with FAISS:

```python
import faiss
import numpy as np

# Build index
d = 256  # descriptor dimension
index = faiss.IndexFlatIP(d)  # inner product (for L2-normalized descriptors)
index.add(database_descs.astype(np.float32))

# Query
distances, indices = index.search(query_descs.astype(np.float32), k=2)
```

FAISS can run on GPU and do ANN search in milliseconds even for million-scale databases. For a visual localization system (not just odometry), this is the retrieval stage and find the k most similar database images to the query, then run LightGlue only on the top candidates.

Code in `descriptor_matching.py` benchmarks brute-force vs FAISS, and shows ratio test vs MNN filtering.

References:

- Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints" (SIFT paper) — IJCV 2004
- https://github.com/facebookresearch/faiss
- Barath et al. "MAGSAC++, a Fast, Reliable and Accurate Robust Estimator" (CVPR 2020)
- https://vnav.mit.edu/lectures.html (Lecture 5)
