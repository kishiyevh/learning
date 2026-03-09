"""
Author: Huseyn Kishiyev
----------------------
Benchmarks different feature matching methods;
  1. Brute-force NN (numpy dot product)
  2. Ratio test filtering
  3. Mutual nearest neighbor (MNN)
  4. FAISS ANN (if installed)

Uses synthetic descriptors (unit sphere random points) to simulate
SuperPoint-style L2-normalized float descriptors.

Run:
  python3 descriptor_matching.py [--n 500] [--dim 256] [--inlier_rate 0.6]

Requires: numpy, matplotlib
Optional: faiss-cpu (pip install faiss-cpu)
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt


def random_unit_descs(n: int, dim: int) -> np.ndarray:
    """Generate n random L2-normalized descriptors in R^dim."""
    d = np.random.randn(n, dim).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d


def simulate_matched_pair(n: int, dim: int, inlier_rate: float, noise_std: float = 0.1):
    """
    Create two descriptor sets with known ground-truth matches.
    inlier_rate: fraction of points that have a true match.
    Returns: desc0, desc1, gt_matches (N,) where gt_matches[i] = j means i->j is correct,
             -1 means no match.
    """
    n_inliers  = int(n * inlier_rate)
    n_outliers = n - n_inliers

    # True matched pairs: desc1_i = desc0_i + small noise
    desc0_inliers = random_unit_descs(n_inliers, dim)
    desc1_inliers = desc0_inliers + np.random.randn(n_inliers, dim).astype(np.float32) * noise_std
    desc1_inliers /= np.linalg.norm(desc1_inliers, axis=1, keepdims=True)

    # Unmatched points in desc1
    desc1_outliers = random_unit_descs(n_outliers, dim)

    # Shuffle desc1
    desc0 = desc0_inliers
    desc1_full = np.vstack([desc1_inliers, desc1_outliers])

    perm = np.random.permutation(n_inliers + n_outliers)
    desc1 = desc1_full[perm]

    # Ground truth: for each i in desc0, which j in desc1 is the true match
    inv_perm = np.argsort(perm)
    gt_matches = np.array([inv_perm[i] for i in range(n_inliers)] + [-1] * n_outliers)

    # Add n_outliers random points to desc0 too (unmatched)
    desc0_outliers = random_unit_descs(n_outliers, dim)
    desc0 = np.vstack([desc0, desc0_outliers])

    return desc0.astype(np.float32), desc1.astype(np.float32), gt_matches


def brute_force_nn(desc0: np.ndarray, desc1: np.ndarray) -> np.ndarray:
    """Nearest neighbor matching using full dot product matrix. Returns matches (N,)."""
    sim = desc0 @ desc1.T  # (N, M)
    return np.argmax(sim, axis=1)


def ratio_test(desc0: np.ndarray, desc1: np.ndarray, ratio: float = 0.8):
    """Return match indices for desc0 that pass Lowe ratio test. -1 if rejected."""
    sim = desc0 @ desc1.T  # (N, M)
    idx_sorted = np.argsort(-sim, axis=1)   # best to worst

    matches = np.full(len(desc0), -1, dtype=np.int32)
    for i in range(len(desc0)):
        j1, j2 = idx_sorted[i, 0], idx_sorted[i, 1]
        # For cosine similarity: lower value = worse match
        # Use distance: dist = 1 - sim (lower = better)
        d1 = 1.0 - sim[i, j1]
        d2 = 1.0 - sim[i, j2]
        if d2 > 1e-6 and d1 / d2 < ratio:
            matches[i] = j1
    return matches


def mutual_nn(desc0: np.ndarray, desc1: np.ndarray):
    """Mutual nearest neighbor matching. Returns (N,) match array, -1 if no mutual match."""
    sim = desc0 @ desc1.T  # (N, M)
    nn_01 = np.argmax(sim, axis=1)   # best match in desc1 for each desc0 point
    nn_10 = np.argmax(sim, axis=0)   # best match in desc0 for each desc1 point

    matches = np.full(len(desc0), -1, dtype=np.int32)
    for i in range(len(desc0)):
        j = nn_01[i]
        if nn_10[j] == i:
            matches[i] = j
    return matches


def precision_recall(matches: np.ndarray, gt_matches: np.ndarray):
    """Compute precision and recall given predicted matches and ground truth."""
    valid    = matches >= 0
    correct  = valid & (matches == gt_matches)
    tp = correct.sum()
    fp = valid.sum() - tp
    fn = (gt_matches >= 0).sum() - tp   # true matches that were missed

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, tp, tp + fp


def run_benchmark(n=500, dim=256, inlier_rate=0.6, noise_std=0.1):
    np.random.seed(42)

    desc0, desc1, gt_matches = simulate_matched_pair(n, dim, inlier_rate, noise_std)
    n_true = (gt_matches >= 0).sum()

    print(f"\nDescriptors: {len(desc0)} x {len(desc1)} | "
          f"dim={dim} | inlier_rate={inlier_rate:.0%} | noise_std={noise_std:.2f}")
    print(f"True matches: {n_true}")
    print(f"\n{'Method':<25} {'Time (ms)':>10} {'Matches':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 75)

    results = {}

    # 1. Brute-force NN
    t0 = time.perf_counter()
    matches_nn = brute_force_nn(desc0, desc1)
    t_nn = (time.perf_counter() - t0) * 1000
    prec_nn, rec_nn, tp_nn, total_nn = precision_recall(matches_nn, gt_matches)
    f1_nn = 2 * prec_nn * rec_nn / (prec_nn + rec_nn + 1e-9)
    results["Brute-force NN"] = (t_nn, total_nn, prec_nn, rec_nn, f1_nn)

    # 2. Ratio test
    t0 = time.perf_counter()
    matches_rt = ratio_test(desc0, desc1, ratio=0.8)
    t_rt = (time.perf_counter() - t0) * 1000
    prec_rt, rec_rt, tp_rt, total_rt = precision_recall(matches_rt, gt_matches)
    f1_rt = 2 * prec_rt * rec_rt / (prec_rt + rec_rt + 1e-9)
    results["Ratio test (0.8)"] = (t_rt, total_rt, prec_rt, rec_rt, f1_rt)

    # 3. MNN
    t0 = time.perf_counter()
    matches_mnn = mutual_nn(desc0, desc1)
    t_mnn = (time.perf_counter() - t0) * 1000
    prec_mnn, rec_mnn, tp_mnn, total_mnn = precision_recall(matches_mnn, gt_matches)
    f1_mnn = 2 * prec_mnn * rec_mnn / (prec_mnn + rec_mnn + 1e-9)
    results["Mutual NN"] = (t_mnn, total_mnn, prec_mnn, rec_mnn, f1_mnn)

    for name, (t, m, prec, rec, f1) in results.items():
        print(f"{name:<25} {t:>10.2f} {m:>8d} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")

    try:
        import faiss
        index = faiss.IndexFlatIP(dim)
        index.add(desc1)
        t0 = time.perf_counter()
        distances, indices = index.search(desc0, k=2)
        t_faiss = (time.perf_counter() - t0) * 1000
        matches_faiss = indices[:, 0]
        # Apply ratio test on FAISS results
        d1 = 1.0 - distances[:, 0]
        d2 = 1.0 - distances[:, 1]
        reject = (d2 < 1e-6) | (d1 / (d2 + 1e-9) >= 0.8)
        matches_faiss_filtered = matches_faiss.copy()
        matches_faiss_filtered[reject] = -1
        prec_f, rec_f, tp_f, total_f = precision_recall(matches_faiss_filtered, gt_matches)
        f1_f = 2 * prec_f * rec_f / (prec_f + rec_f + 1e-9)
        print(f"{'FAISS + ratio (0.8)':<25} {t_faiss:>10.2f} {total_f:>8d} "
              f"{prec_f:>10.3f} {rec_f:>8.3f} {f1_f:>8.3f}")
    except ImportError:
        print("FAISS not installed — skipping (pip install faiss-cpu)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",            type=int,   default=500)
    parser.add_argument("--dim",          type=int,   default=256)
    parser.add_argument("--inlier_rate",  type=float, default=0.6)
    parser.add_argument("--noise_std",    type=float, default=0.1)
    args = parser.parse_args()

    run_benchmark(args.n, args.dim, args.inlier_rate, args.noise_std)
