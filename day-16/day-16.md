# March 3, 2026

## ONNX Runtime Performance Tuning and Pipeline Profiling

Went through the ONNX Runtime performance documentation and did actual profiling of the SuperPoint + LightGlue pipeline. The GTX 1650 Ti has 4GB VRAM and compute capability 7.5 (Turing architecture), which supports cuDNN but not all TensorRT optimizations.

### Profiling first, optimizing second

The first thing to do before any optimization is measure. ORT has built-in profiling:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.enable_profiling = True
opts.profile_file_prefix = "ort_profile"  # writes ort_profile_TIMESTAMP.json

session = ort.InferenceSession("superpoint.onnx", sess_options=opts)

# Run inference
session.run(...)

# Flush profile
profile_path = session.end_profiling()
print(f"Profile saved to: {profile_path}")
```

The JSON file can be loaded in Chrome's `chrome://tracing` or in Perfetto (https://ui.perfetto.dev) to see a flame chart of operator execution times.

### What I found in the SuperPoint profile

For a 640x480 input on the GTX 1650 Ti with `CUDAExecutionProvider`:

```
Conv_0:     1.2 ms   (encoder first conv)
Conv_1-8:   4.3 ms   (VGG encoder blocks)
detector_head: 0.8 ms
descriptor_head: 1.1 ms
softmax + NMS: 0.6 ms
TOTAL:      ~8 ms
```

The bulk of the time is in the encoder. The detector and descriptor heads together are only about 25% of total inference time. This means improving the encoder would have the biggest impact, options are quantization or smaller image input.

### LightGlue profile

LightGlue is more complex because of the attention layers:

```
Cross-attention layer 0: 3.1 ms
Cross-attention layer 1: 3.0 ms
...
Cross-attention layer 8: 3.2 ms
Sinkhorn matching:       2.1 ms
TOTAL (9 layers):       ~30 ms
```

So LightGlue takes 3-4x longer than SuperPoint. With 300-400 keypoints per image, the attention is O(N^2). At 1000 keypoints, it would be ~4x slower still. Keeping keypoint count in the 200-400 range is important for this GPU.

### Key ORT optimization flags

Graph optimization level. This is the first thing to enable:

```python
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

This enables constant folding, CSE (common subexpression elimination), and layout optimization for conv operators.

Saving the optimized model to disk avoids the optimization cost on subsequent runs:

```python
opts.optimized_model_filepath = "superpoint_optimized.onnx"
```

For CUDA provider, the `cudnn_conv_algo_search` setting matters:

```python
cuda_opts = {
    "cudnn_conv_algo_search": "EXHAUSTIVE",   # finds best algorithm once at first run
    # or "HEURISTIC" for faster startup but potentially slower inference
}
```

`EXHAUSTIVE` runs all available cuDNN algorithms on the first session run and caches the result. First run takes several seconds extra; subsequent runs are faster.

### FP16 inference

The GTX 1650 Ti supports FP16 compute (Turing has tensor cores). ORT does not auto-convert to FP16 for CUDA EP, but we can convert the model first:

```bash
# Using onnxconverter-common
pip install onnxconverter-common
python3 -c "
from onnxconverter_common import float16
import onnx
model = onnx.load('superpoint.onnx')
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, 'superpoint_fp16.onnx')
"
```

In testing on the 1650 Ti, FP16 SuperPoint reduced inference from ~8 ms to ~5 ms. LightGlue FP16 went from ~30 ms to ~19 ms. Total pipeline: ~24 ms (~42 FPS theoretical), compared to ~38 ms (~26 FPS) with FP32.

### CPU Overhead

One thing the profiler revealed is that CPU-GPU data transfer and Python overhead can dominate at small batch sizes. The CUDA kernel itself might take 8 ms, but the total wall-clock time is 11 ms because of synchronization and data marshaling.

IO binding (discussed Feb 25) helps, but the bigger gain comes from batching. If you are processing a video stream, you can batch two frames together for SuperPoint (batch size 2 instead of running it twice). LightGlue's cross-image attention cannot be trivially batched, but the SuperPoint encoder can.

### Target pipeline latency for visual odometry

For a real-time drone system, the VIO front-end needs to run at >= 20 Hz (50 ms budget per frame pair). On the GTX 1650 Ti with FP16 ONNX models:

- SuperPoint x2: ~10 ms
- LightGlue: ~19 ms
- OpenCV geometry (essential matrix + pose recovery): ~2 ms
- Total: ~31 ms = ~32 FPS

This is within budget, with some headroom for preprocessing and topic publishing.

References:

- https://onnxruntime.ai/docs/performance/
- https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://github.com/fabio-sim/LightGlue-ONNX
