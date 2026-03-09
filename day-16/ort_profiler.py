"""
Author: Huseyn Kishiyev
---------------
Profiles ONNX Runtime inference for any model.
Reports per-operator latency, total latency stats, and throughput.
Optionally compares FP32 vs FP16 models if both are provided.

Usage:
  python3 ort_profiler.py --model superpoint.onnx [--model_fp16 superpoint_fp16.onnx] \
      --input_shape 1 1 480 640 [--gpu] [--warmup 10] [--runs 100]

For LightGlue (4 inputs, one per arg):
  python3 ort_profiler.py --model lightglue.onnx \
      --input_shape 1 256 2  1 256 2  1 256 256  1 256 256 [--gpu]

Requires: onnxruntime (or onnxruntime-gpu), numpy
"""

import argparse
import time
import json
import os
import tempfile
import numpy as np


def create_dummy_input(session) -> dict:
    """Create random dummy inputs matching the model's expected types and shapes."""
    inputs = {}
    for inp in session.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(1)  # fill dynamic dims with 1
        dtype_map = {
            "float": np.float32,
            "double": np.float64,
            "int32": np.int32,
            "int64": np.int64,
        }
        np_dtype = dtype_map.get(inp.type.replace("tensor(", "").replace(")", ""), np.float32)
        inputs[inp.name] = np.random.rand(*shape).astype(np_dtype)
    return inputs


def run_benchmark(session, inputs: dict, n_runs: int = 50):
    """Run inference n_runs times, return latency stats in ms."""
    output_names = [o.name for o in session.get_outputs()]

    # Warmup
    for _ in range(5):
        session.run(output_names, inputs)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(output_names, inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    return np.array(latencies)


def parse_ort_profile(profile_json: str) -> list[dict]:
    """Parse ORT profiling JSON and return top operators by duration."""
    with open(profile_json) as f:
        data = json.load(f)

    ops = {}
    for event in data:
        if event.get("cat") == "Node":
            name = event.get("name", "unknown")
            dur  = event.get("dur", 0) / 1000.0  # microseconds -> ms
            if name not in ops:
                ops[name] = {"count": 0, "total_ms": 0.0}
            ops[name]["count"]    += 1
            ops[name]["total_ms"] += dur

    sorted_ops = sorted(ops.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    return sorted_ops[:20]  # top 20 by time


def benchmark_model(model_path: str, providers: list, n_runs: int = 50,
                    enable_profiling: bool = False, label: str = ""):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    profile_path = None
    if enable_profiling:
        tmpdir = tempfile.mkdtemp()
        opts.enable_profiling   = True
        opts.profile_file_prefix = os.path.join(tmpdir, "ort_prof")

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    inputs  = create_dummy_input(session)

    print(f"\n{'='*60}")
    print(f"Model: {os.path.basename(model_path)}  ({label})")
    print(f"Provider: {session.get_providers()[0]}")

    input_info = [(i.name, i.shape, i.type) for i in session.get_inputs()]
    for name, shape, dtype in input_info:
        print(f"  Input: {name}  shape={shape}  type={dtype}")

    latencies = run_benchmark(session, inputs, n_runs)

    print(f"\nLatency over {n_runs} runs:")
    print(f"  Mean:   {latencies.mean():.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  Min:    {latencies.min():.2f} ms")
    print(f"  Max:    {latencies.max():.2f} ms")
    print(f"  Throughput: {1000.0 / latencies.mean():.1f} inferences/sec")

    if enable_profiling:
        prof_file = session.end_profiling()
        top_ops = parse_ort_profile(prof_file)
        print(f"\nTop operators by total time:")
        for op_name, info in top_ops[:10]:
            print(f"  {op_name:<50} {info['total_ms']:>8.3f} ms  (x{info['count']})")
        print(f"\nFull profile at: {prof_file}")
        print("View in Chrome at chrome://tracing  or  https://ui.perfetto.dev")

    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True, help="Path to ONNX model (FP32)")
    parser.add_argument("--model_fp16",   default=None,  help="Optional FP16 ONNX model to compare")
    parser.add_argument("--gpu",          action="store_true")
    parser.add_argument("--warmup",       type=int, default=5)
    parser.add_argument("--runs",         type=int, default=50)
    parser.add_argument("--profile",      action="store_true", help="Enable ORT operator profiling")
    args = parser.parse_args()

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.gpu else ["CPUExecutionProvider"])

    lat_fp32 = benchmark_model(
        args.model, providers, args.runs, args.profile, label="FP32"
    )

    if args.model_fp16:
        lat_fp16 = benchmark_model(
            args.model_fp16, providers, args.runs, args.profile, label="FP16"
        )
        speedup = lat_fp32.mean() / lat_fp16.mean()
        print(f"\nSpeedup FP16 vs FP32: {speedup:.2f}x")


if __name__ == "__main__":
    main()
