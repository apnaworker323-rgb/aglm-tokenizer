# AGLM Identical-Data GPU Training Benchmark

Status: **PASSED**  
Dataset manifest: `/run/media/akash/18FAA791FAA76A28/aglm_project/aglm_tokenized_dataset/metadata/dataset_manifest.json`  
Input schedule SHA256 (identical in both runs): `59559e35e5287d4fd380fb3ef321196a6c1a2c86991a00fc4d0ba462507a4125`

## Result

| Input path | tokens/s | effective raw MiB/s | p50 step ms | p95 step ms | peak allocated MiB | mean GPU util |
|---|---:|---:|---:|---:|---:|---:|
| Resident bounded schedule | 9379.5 | 0.044 | 106.02 | 129.25 | 2724.4 | 96.70212765957447 |
| uint32 mmap shards | 9260.2 | 0.043 | 106.76 | 130.38 | 2723.5 | 97.82978723404256 |

Mmap end-to-end throughput difference from resident compute ceiling: **1.27%**.

## Interpretation

- The canonical mmap path is within **1.27%** of the bounded resident-data compute ceiling, so shard I/O is not the limiting component in this run.
- Dataset conversion ran **200.5x faster** in raw-byte terms than this GPU training benchmark. Pre-tokenization can therefore stay comfortably ahead of this exact benchmark workload.
- At the measured mmap rate, one pass over all 1,479,404,911 token positions would take about **44.38 hours**. This is a projection for this compact benchmark model and configuration, not a final-model training estimate.

## Experimental contract

- Both runs start from the same deterministic seed and consume the exact same input/target batch schedule.
- No token is truncated, cast to uint16, clamped, or reduced modulo another vocabulary.
- The input embedding directly addresses all 1,551,017 AGLM IDs.
- The loss is normalized hierarchical adaptive softmax over all 1,551,017 targets. It is memory-safe on this GPU, but is not a flat dense-softmax parameterization.
- This is a bounded forward + backward + gradient clipping + AdamW benchmark. It is not a full training run and writes no checkpoint.
- Effective raw MiB/s uses the completed corpus manifest's measured `4.914080` raw bytes/token.

## Hardware and model

- GPU: NVIDIA GeForce RTX 3050 (5804 MiB)
- PyTorch/CUDA: 2.12.0+cu130 / 13.0
- Precision: bf16
- Shape: batch=2, seq_len=512, d_model=192, layers=4, heads=6
- Trainable parameters: 67,872,091
- Warmup/timed steps per path: 10 / 100
