# Native performance options for the AoE2 agent

Research date: 2026-10-16

Keep Python for orchestration. Fix the measured algorithmic scaling problem and
unnecessary image conversions first; consider one native component only if
profiling still identifies a material bottleneck. This recommendation combines
local inspection with five primary sources; it does not establish a Rust/C++
speedup or Windows VM capture compatibility.

## What the local evidence supports

[Temporal association](../../apps/agent/src/temporal_entities.py) compares old/new
entities and then scans the entire candidate list twice for each candidate.
Those ambiguity checks are O(C²) in candidate count. The call runs synchronously
in [the observation loop](../../apps/agent/src/loops/source.py). Grouping costs by
old/new entity and pruning spatially impossible pairs should precede a port.

A synthetic diagnostic timed the second observation with unchanged camera:
median of three repetitions, fresh tracker each time, random 1920×1080 RGB
(NumPy seed 42), 64×64 villager boxes in 12 columns, 110-pixel versus 3-pixel
spacing. For 20/40/60 entities, separated scenes took 4.07/7.33/11.93 ms;
near-overlapping scenes took 14.39/224.17/1064.86 ms, with all matches ambiguous.
This is an adversarial scaling diagnostic, **not representative gameplay timing**.

The [recorded detector baseline](../runbooks/detector-quality-baseline.md) is
109.5 ms median, 147.2 ms p95 over 32 decoded images on this Mac's CPU provider.
It excludes capture, JPEG decoding, transport, tracking, and scheduling. The
[full Windows/Mac stream](../streaming-perception.md) remains unmeasured.

## What native code would change

ONNX Runtime's Python binding already invokes its native inference session and
releases the GIL around `Run()`. Moving that call to C++ or Rust leaves the
model's computation in ONNX Runtime; benefit would need to come from surrounding
work, copies, or a different execution configuration. [Official binding
source](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/python/onnxruntime_pybind_state.cc).

On Apple Silicon, benchmark CoreML through the existing Python session. Official
documentation provides Python provider options for CPU, GPU, and Neural Engine.
It warns that dynamic shapes can hurt performance and enabling ANE does not
guarantee the entire model runs there. Supported operators and actual placement
matter. Measure warm inference separately from model compilation; caching can
reduce repeated compilation but requires correct invalidation when weights
change. Validate detections as well as latency. [CoreML execution
provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).

Many NumPy operations already run in low-level code and release the GIL; object
arrays are an exception. Vectorize substantial Python loops before introducing
an extension, and measure remaining allocation/copy costs. Threads require
care with shared mutable arrays. [NumPy thread
safety](https://numpy.org/doc/stable/reference/thread_safety.html).

[Capture code](../../apps/agent/src/screen.py) creates MSS contexts and performs
image reconstruction/encoding; the HUD path also encodes PNG before downstream
decoding. Reduce avoidable work first. If capture remains dominant, compare MSS
against a small C++ Desktop Duplication backend. Microsoft's API delivers DXGI
surfaces and changed-region metadata, enabling GPU image processing. That is an
API capability, not evidence that this VM supports it or that CPU readback and
encoding become faster. [Desktop Duplication
API](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api).

For a persistent CPU hotspot, prefer one coarse Rust extension using PyO3 and
maturin, with Python retaining orchestration. PyO3 documents this packaging path
and requires an optimized release build for performance comparisons. C++ is a
reasonable choice for a Windows graphics component because Microsoft's example
already uses its graphics interfaces. These are integration recommendations,
not claims that either language is intrinsically faster here. [PyO3
guide](https://pyo3.rs/main/).

Accept a native component only after equivalent-output replay shows lower
end-to-end p50/p95 latency including boundary copies, with enough benefit to
justify its build and deployment costs.
