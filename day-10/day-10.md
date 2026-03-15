# LightGlue-ONNX, Architecture and the ONNX Runtime Pipeline

Spent the day on the LightGlue-ONNX repo (https://github.com/fabio-sim/LightGlue-ONNX) and the ONNX Runtime performance documentation. The goal was to understand what LightGlue actually does differently from brute-force nearest-neighbor matching, and what the ONNX conversion pipeline looks like.

### What LightGlue is

LightGlue (Lindenberger et al., 2023) is a learned feature matcher that takes two sets of keypoints + descriptors and produces sparse correspondences. It is the successor to SuperGlue but significantly faster at inference time.

The important architectural ideas,

**Self- and cross-attention transformer** — Each keypoint's descriptor is updated by attending to other keypoints in the same image (self-attention, captures local context) and to keypoints in the other image (cross-attention, finds matches). This is done iteratively across multiple "layers" (default 9 in the full model).

**Adaptive early exit** — LightGlue predicts per-layer confidence in its assignment. If it is already confident enough after layer 3 or 4, it stops early. This is why LightGlue is faster than SuperGlue in practice — easy matching pairs exit early, hard ones go all the way through.

**Soft assignment with dustbin** — Like SuperGlue, it uses the Sinkhorn algorithm to solve an optimal transport problem on the assignment matrix, with a dustbin column for unmatched keypoints.

The output is a list of (index_in_image_A, index_in_image_B) matched pairs with confidence scores.

### ONNX conversion pipeline

The fabio-sim repo exports both SuperPoint and LightGlue to ONNX using `torch.onnx.export`. The key challenge is that LightGlue has dynamic shapes (variable number of keypoints N, different for each image pair) and the optional early exit (dynamic control flow).

```python
# Simplified export call (from the repo's export script)
torch.onnx.export(
    model,
    (kpts0, kpts1, desc0, desc1),
    "lightglue.onnx",
    opset_version=17,
    input_names=["kpts0", "kpts1", "desc0", "desc1"],
    output_names=["matches0", "matches1", "mscores0", "mscores1"],
    dynamic_axes={
        "kpts0":  {0: "batch", 1: "num_kpts0"},
        "kpts1":  {0: "batch", 1: "num_kpts1"},
        "desc0":  {0: "batch", 1: "num_kpts0"},
        "desc1":  {0: "batch", 1: "num_kpts1"},
    }
)
```

The dynamic axes are critical. Without them, the ONNX model would only accept the exact input sizes used during export.

The early exit is handled by either removing it for the ONNX version (fixed number of layers) or using ONNX's `If` operator for conditional execution. The fabio-sim implementation disables adaptive depth for the ONNX export and fixes the number of layers, which simplifies the graph.

### ONNX Runtime execution providers

ONNX Runtime (ORT) can run models on CPU, CUDA, TensorRT, or other backends via "execution providers." On the GTX 1650 Ti:

- `CUDAExecutionProvider` uses cuDNN for attention operations — this is the right choice for LightGlue
- `TensorRTExecutionProvider` builds an optimized TRT engine — faster but requires TRT installed and has a slow first-run build time (minutes)
- `CPUExecutionProvider` — baseline, no GPU

For the GTX 1650 Ti with 4GB VRAM, TensorRT is marginal — the LightGlue model with default settings fits, but you lose VRAM budget for the rest of the pipeline. Using `CUDAExecutionProvider` with fp16 mode enabled gives a good tradeoff.

### ORT session options for performance

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4  # for CPU parallelism

providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 2 * 1024**3,   # 2 GB cap to leave room for other ops
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
    }),
    "CPUExecutionProvider"
]

session = ort.InferenceSession("lightglue.onnx", sess_options=opts, providers=providers)
```

### IO binding for zero-copy GPU inference

When inputs are already on GPU (as CUDA tensors), copying them to CPU for ORT and back to GPU wastes time. ORT's IO binding avoids this:

```python
binding = session.io_binding()
binding.bind_input(
    name="kpts0",
    device_type="cuda",
    device_id=0,
    element_type=np.float32,
    shape=kpts0.shape,
    buffer_ptr=kpts0.data_ptr()   # if using PyTorch tensor
)
session.run_with_iobinding(binding)
```

This is important when SuperPoint (also GPU) feeds directly into LightGlue — you avoid two unnecessary CPU-GPU round trips.

### What the full pipeline looks like

```
Image A (GPU)                   Image B (GPU)
    |                               |
SuperPoint (ONNX, CUDA)        SuperPoint (ONNX, CUDA)
    |                               |
kpts_A, desc_A                 kpts_B, desc_B
    \                              /
     LightGlue (ONNX, CUDA)
          |
    matched indices + scores
          |
    Essential matrix estimation (OpenCV / MAGSAC)
          |
    Relative pose (R, t)
```

Total pipeline time on GTX 1650 Ti for 640x480 images: roughly 40-60 ms with CUDAExecutionProvider.

References:

- https://github.com/fabio-sim/LightGlue-ONNX
- https://onnxruntime.ai/docs/performance/
- Lindenberger et al., "LightGlue: Local Feature Matching at Light Speed" (2023) — arXiv:2306.13643
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
