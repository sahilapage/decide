# CPU latency benchmarks

Latency is 20% of the score: full marks at p95 ≤ 25 ms, zero at 100 ms, and a
mean p95 over 100 ms fails the gate. The judges measure on a 4-core Linux box we
have not seen, so we benchmark two layouts:

* **4phys**: four distinct physical cores (`taskset -c 0,2,4,6`)
* **2phys_4vcpu**: four logical CPUs on two physical cores (`taskset -c 0-3`),
  which is what a typical "4 vCPU" cloud VM gives you

Note that `taskset -c 0-3` on an SMT machine is only **two** physical cores.
Check with `lscpu -e=CPU,CORE`.

## How to reproduce

```bash
# int8 copy of a packed student (dynamic quantisation, no retraining)
mkdir -p exports/opt && cat artifacts/student/model.onnx.part* > exports/opt/model.onnx
cp artifacts/student/{tokenizer.json,meta.json,policy.json} exports/opt/
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; \
quantize_dynamic('exports/opt/model.onnx','exports/opt/model.int8.onnx',weight_type=QuantType.QInt8)"

python tools/latency_sweep.py exports/opt sweep.csv --reps 3
```

`bench_cpu.py` times the model in-process (tokenise + ONNX run), one request at
a time after warm-up. HTTP adds roughly 3 ms on top (measured with
`selfcheck.py`).

## Ettin-32M student (`artifacts/student`, as shipped at 248a2bd)

fp32 ONNX is 128 MB, dynamic int8 is 32.5 MB. Machine: AMD Ryzen 7 7735HS
laptop (8 cores / 16 threads), 300 dev tickets × 3 decisions per run, 3 runs per
config. Columns: median over runs of the mean-of-decisions p95, the min–max of
that p95 across runs, and the median p50.

| Layout | Model | Threads | Spin | p95 median | p95 min–max | p50 |
|---|---|---|---|---|---|---|
| 4phys | fp32 (shipped) | 4 | on | 29.6 | 13.5–56.0 | 9.5 |
| 4phys | int8 | 4 | on | **8.7** | 8.1–56.8 | **6.4** |
| 4phys | int8 | 4 | off | 10.4 | 9.6–29.2 | 6.9 |
| 4phys | int8 | 2 | off | 31.4 | 24.9–34.8 | 8.7 |
| 4phys | int8 | 1 | off | 42.0 | 36.6–46.9 | 12.3 |
| 2phys_4vcpu | fp32 (shipped) | 4 | on | 19.3 | 16.9–23.1 | 13.3 |
| 2phys_4vcpu | int8 | 4 | on | 12.0 | 11.9–96.5 | 9.4 |
| 2phys_4vcpu | int8 | 4 | off | **11.7** | 11.3–21.8 | 9.0 |
| 2phys_4vcpu | int8 | 2 | off | 14.2 | 10.6–26.0 | 8.4 |
| 2phys_4vcpu | int8 | 1 | off | 16.0 | 15.3–91.0 | 12.0 |

### What we conclude

* **int8 is reliably faster**: p50 drops about 1.4× in both layouts.
* **int8 + 4 threads** is fastest or tied in both layouts.
* **Spinning off** costs ~0.5 ms at p50 but removed the 56–96 ms outlier runs
  seen with spinning on. Current candidate: int8, 4 threads, `DECIDE_SPIN=0`.
* **Laptop p95 is noisy** (the same config ranges 3–8× across runs), so p50 is
  the trustworthy number here. Confirm the final config on a quieter desktop
  CPU before changing what ships.

### int8 accuracy check (done) — we ship fp32

Full dev, every call through the CPU runtime, policy refitted on each
variant's own logits and cross-fitted (`tools/quant_eval.py`, Ettin-32M short
prompts, i9-13900K, 2 physical cores):

| Variant | p95 | cross-fitted dev total |
|---|---|---|
| fp32 | 14.2 ms | **83.4** |
| int8 dynamic (default) | 7.9 ms | 80.4 |
| int8 MatMul-only, per-channel | 7.9 ms | 80.1 |
| int8 MatMul-only, per-tensor | 7.8 ms | 80.0 |
| int8 per-channel, reduce_range | 8.0 ms | 79.9 |

int8 costs ~3 points, almost all on **grounded** (accuracy 99.1% → 92.0%):
spotting "15 minutes" vs "30 minutes" needs precise activations. Latency is
only worth the trade if fp32 p95 is above ~40 ms, so the answer is a smaller
fp32 model instead.

### Ettin-17M student (shipped, fp32 68 MB)

| Machine / layout | Threads | p95 (mean of decisions) |
|---|---|---|
| i9-13900K, 2 physical cores (`taskset -c 0-3`), in-process | 4 | 5.4 ms |
| i9-13900K, 2 physical cores, in-process | 2 | 5.0 ms |
| Apple M5, HTTP via `selfcheck.py` (full dev, 4,500 calls) | 4 | 2.5 ms |

Ettin-32M fp32 on the same i9 layout is 13.4 ms; the 17M matches its quality
(cross-fitted out-of-fold 84.1 vs 83.8), so the 17M ships with ~5x headroom
under the 25 ms full-marks line.
