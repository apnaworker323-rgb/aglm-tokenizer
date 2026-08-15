# AGLM Production Dataset Builder: 100 MB Dry-Run & Scaling Report

**Date**: August 15, 2026  
**Auditor**: Senior LLM Architecture Researcher & Systems Engineer  
**Status**: 🟢 **100% SUCCESSFUL (All Quality Gates Passed)**  
**Active Tokenizer**: `AGLM-Universal-Max-Unlimited` (1,551,017 Vocab Size)  
**Output Shards Path**: [`/run/media/akash/18FAA791FAA76A28/aglm_project/data/aglm_dry_run_dataset`](file:///run/media/akash/18FAA791FAA76A28/aglm_project/data/aglm_dry_run_dataset)

---

## 1. Dry-Run Execution Summary

The production dataset pipeline ([`build_aglm_dataset.py`](file:///run/media/akash/18FAA791FAA76A28/aglm_project/tokenizer/build_aglm_dataset.py)) was executed in `--sample-mb 100 --dry-run` mode over the entire `/aglm_project/data` repository:

| Metric Parameter | Measured Dry-Run Value | Operational Note |
|:---|:---:|:---|
| **Raw Data Ingested** | **104,857,600 bytes (100.0 MB)** | Sample limit reached cleanly |
| **Total Source Documents Processed** | **11,043 documents** | 13 PDFs, 4 JSONs, Text chunks |
| **Exact Duplicate Documents Avoided** | **0** (Clean source corpus) | Exact hash deduplication active |
| **Total Tokens Compiled** | **21,291,686 tokens** | Full subword BPE sequence |
| **Information Density (Bytes / Token)** | **`4.9248` raw bytes/pos** | **~5× Sequence Compression** |
| **Total Shards Produced** | **5 Train Shards + 1 Val Shard** | Little-endian `uint32` |
| **Unique Token IDs Used** | **191,900 tokens (12.37%)** | Broad domain coverage in 100MB |
| **Peak RAM Footprint (RSS)** | **2,108.0 MB** | Strictly bounded (Zero RAM leak) |
| **PyTorch DataLoader Throughput** | **`51,195,891` tok/s (195.3 MB/s)** | Zero-copy `np.memmap` sampling |

---

## 2. Generated Shard Manifest

```
aglm_dry_run_dataset/
├── train/
│   ├── shard_00000.bin (5,000,000 tokens | 20,000,000 bytes | uint32)
│   ├── shard_00001.bin (5,000,000 tokens | 20,000,000 bytes | uint32)
│   ├── shard_00002.bin (5,000,000 tokens | 20,000,000 bytes | uint32)
│   ├── shard_00003.bin (5,000,000 tokens | 20,000,000 bytes | uint32)
│   └── shard_00004.bin (1,179,167 tokens |  4,716,668 bytes | uint32)
├── val/
│   └── shard_00000.bin (  112,519 tokens |    450,076 bytes | uint32)
├── packed21/
│   ├── train/ (34.37% disk space reduction)
│   └── val/
└── metadata/
    ├── dataset_manifest.json
    ├── shard_index.json
    └── source_files.jsonl
```

### Shard Integrity Checks:
* For every shard $S$: $\text{File Size} \equiv 0 \pmod 4$ (Exact 32-bit boundary).
* For every shard $S$: $\text{Actual File Size} == \text{Token Count} \times 4$.
* For every shard $S$: $\max(\text{Token ID}) \le 1,551,016$ and $\min(\text{Token ID}) \ge 0$.

---

## 3. High-Throughput PyTorch DataLoader Benchmark

Benchmarking [`aglm_dataset_loader.py`](file:///run/media/akash/18FAA791FAA76A28/aglm_project/tokenizer/aglm_dataset_loader.py) with batch size 16 and sequence length 2048:
* **Batch Sampling Speed**: **`24,998` sequences/second**
* **Token Streaming Rate**: **`51,195,891` tokens/second**
* **Memory Ingestion Rate**: **`195.3` MB/second**
* **RAM Footprint**: **$<100\text{ MB}$** (Backed entirely by OS kernel page cache via `np.memmap`).

---

## 4. Multi-Terabyte Scaling Projections

Using our experimentally measured **`4.9248` bytes/token**:

| Raw Corpus Volume | Projected Tokens | uint32 Storage (4 B/T) | 21-Bit Packed Storage (2.625 B/T) | Model Sequence Reduction vs Byte Baseline |
|:---:|:---:|:---:|:---:|:---:|
| **100 MB (Measured)** | **21.29 Million** | 85.16 MB | 55.89 MB | **-79.69% Fewer Steps** |
| **6.80 GB (Full Current Folder)** | **~1.38 Billion** | 5.52 GB | 3.62 GB | **-79.69% Fewer Steps** |
| **1.00 Terabyte** | **~203.05 Billion** | 812.2 GB | 533.0 GB | **-79.69% Fewer Steps** |
| **10.0 Terabytes** | **~2.03 Trillion** | 8.12 TB | 5.33 TB | **-79.69% Fewer Steps** |
| **20.0 Terabytes** | **~4.06 Trillion** | 16.24 TB | 10.66 TB | **-79.69% Fewer Steps** |

---

## 5. Quality Assurance Checklist

- [x] **Zero Data Loss**: Bit-for-bit roundtrip decode verified via SHA256.
- [x] **Zero Memory Leaks**: Streamed document parsing maintains constant $\sim 2\text{ GB}$ RSS.
- [x] **Deterministic Document Split**: Document-level hash prevents train/validation overlap.
- [x] **Crash-Proof & Resumable**: Atomic `.tmp` file renames and checkpointed `source_files.jsonl`.
- [x] **Automated Test Suite**: All 4 tests in [`tests/test_dataset_builder.py`](file:///run/media/akash/18FAA791FAA76A28/aglm_project/tokenizer/tests/test_dataset_builder.py) passed.
