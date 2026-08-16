# AGLM Tokenizer Performance and Exactness Report

Generated: 2026-08-15

## Decision

**Performance gate passed. Full-corpus conversion remains NOT STARTED.**

- Exact 1 GiB Python reference: **0.447 MiB/s**.
- Exact Rust accelerator: **10.654 MiB/s** with one thread and **35.420 MiB/s** with eight threads.
- Complete bounded dataset pipeline: **12.286 MiB/s raw** over 117,353,781 bytes, including extraction, exact dedupe, token frequency accounting, uint32 writes, fsync, SHA256, manifests, and final verification.
- Every 1 GiB native mode produced the same ordered uint32 digest as the Python reference: `914262e81c4f482d4be3c03a2e74a65658b56351cdfb72e755c1715ec3d23596`.
- The requested 10–20 MiB/s floor is cleared before any large conversion is authorized.

Machine: Intel Core i7-6700, 4 physical cores / 8 logical CPUs, 31 GiB RAM, little-endian x86_64, Linux 7.0.0-29, Python 3.14.4. The verified runtime is `regex==2026.5.9` with Unicode 17.0 tables.

## Exact 1 GiB in-memory benchmark

The same 1,073,741,824-byte FineWeb sample was read once, decoded, and retained in RAM. Timed tokenizer modes did not read the corpus from disk. Validation SHA256 was performed after native timing.

| Backend / mode | Workers | Seconds | Raw MiB/s | Tokens/s | Tokens |
|---|---:|---:|---:|---:|---:|
| Python reference, single | 1 | 2,289.672 | 0.447 | 94,089 | 215,432,465 |
| Native, single | 1 | 96.111 | 10.654 | 2,241,496 | 215,432,465 |
| Native, threads | 2 | 55.205 | 18.549 | 3,902,440 | 215,432,465 |
| Native, processes | 2 | 60.505 | 16.924 | 3,560,553 | 215,432,465 |
| Native, threads | 4 | 34.846 | 29.386 | 6,182,333 | 215,432,465 |
| Native, processes | 4 | 37.728 | 27.142 | 5,710,165 | 215,432,465 |
| Native, threads | 8 | 28.910 | 35.420 | 7,451,802 | 215,432,465 |
| Native, processes | 8 | 31.356 | 32.657 | 6,870,576 | 215,432,465 |

One native thread is 23.82× faster than the reference. Eight shared-trie threads are 79.20× faster. Threads beat processes at every equal native worker count because the Rust code releases the GIL and avoids returning uint32 data through process IPC.

## Wall-clock stage decomposition

These are isolated measurements, so rows with different byte bases are not additive. Disk and UTF-8 rows use 1 GiB. Parser/tokenizer rows use 4 MiB. Write and shard-SHA rows use the 3,353,800-byte uint32 result.

| Requested bucket | Seconds | Throughput | Finding |
|---|---:|---:|---|
| Disk read | 4.007 | 255.583 MiB/s raw | Not a bottleneck |
| UTF-8 parsing | 1.967 | 520.653 MiB/s raw | Not a bottleneck |
| Text extraction / metadata | 0.0049 | 824.121 MiB/s raw | Not a bottleneck for plain text |
| Dedup SHA256 | 0.0141 | 283.326 MiB/s raw | Not a bottleneck |
| Regex pre-tokenization | 1.149 | 3.481 MiB/s raw | Secondary reference hotspot |
| Trie/BPE lookup + DP/backtracking | 6.064 | 0.660 MiB/s raw | Dominant reference hotspot |
| Traditional merge operations | 0 | n/a | Runtime uses no ranked pair-merge loop |
| Python IDs → little-endian uint32 | 0.376 | 10.644 MiB/s raw | Material reference overhead; removed from native path |
| Shard write + flush + fsync | 0.134 | 23.792 MiB/s output | Material after acceleration |
| Shard SHA256 readback | 0.0080 | 402.124 MiB/s output | Not a bottleneck |
| Multiprocessing IPC only, 2 workers | 0.138 | 29.089 MiB/s one-way raw | Below memory/disk; avoidable for native threads |

The algorithm documented as “BPE” is actually regex segmentation followed by enumeration of every vocabulary-trie prefix at each byte position and a minimum-token shortest-path dynamic program. The corresponding “merge operations” count is therefore exactly zero. On a 256 KiB diagnostic prefix there were 262,144 reachable byte positions, 893,321 trie matches, and 263,125 successful DP relaxations.

## Python allocation and hotspot evidence

`cProfile` on 4 MiB recorded 27,987,437 calls in 16.208 profiler-seconds:

| Function | Calls | Own seconds | Cumulative seconds |
|---|---:|---:|---:|
| `BPEEngine.encode` | 64 | 0.738 | 16.208 |
| `BPEEngine.encode_segment` | 810,785 | 4.415 | 13.656 |
| `ByteTrie.all_prefix_matches` | 4,102,537 | 6.688 | 8.912 |
| Python `list.append` | 14,825,631 | 1.943 | 1.943 |
| `ScriptSegmenter.pre_tokenize` | 64 | 1.300 | 1.484 |

`tracemalloc` on 1 MiB retained 1.69 MiB of token-list objects and reached a 2.31 MiB traced peak; instrumentation increased that run to 3.261 s, so it is evidence of allocation sites rather than a throughput number.

Linux `perf` on the live reference run attributed 47.47% of cycles to Python frame evaluation, plus dictionary lookup, tuple construction, deallocation, and regex property matching. Native `perf` attributed 33.13% to `NativeBpe::encode_one`, 10.22% to forward/reverse regex-automata search, 4.79% to the Unicode-version risk scan, and only 5.91% to Python frame evaluation. `py-spy` was not installed; no dependency was added merely for profiling.

## Native implementation

The source is in `native/aglm_native/` and is exposed by `aglm_tokenizer/native.py`. It preserves the reference algorithm rather than substituting greedy BPE:

- compact 4,696,845-node / 4,696,844-edge byte trie;
- constant-time root-byte transitions;
- a provably exact whole-segment terminal fast path;
- reusable shortest-path DP buffers and strict reference tie-breaking;
- linear-time Rust Unicode regex plus an exact repair of the reference negative-lookahead whitespace rule;
- direct little-endian uint32 byte output, avoiding Python integer/list and NumPy-conversion overhead;
- PyO3 GIL release, allowing one immutable trie to be shared by threads;
- exact dependency pins and a little-endian build guard.

### Unicode-table safety

The Python `regex` oracle uses Unicode 17.0 while pinned Rust `regex-syntax` uses Unicode 16.0. An audit of every boundary-relevant general category, script, whitespace, and contraction case-fold property found 4,734 differing scalars. Any document containing one is routed in full to the Python reference and counted in dataset statistics. Changes to `Extended_Pictographic` are excluded from that set because the earlier punctuation alternative shadows the later emoji alternative.

Only 7 of 16,385 chunks in the 1 GiB test required fallback: 458,752 raw bytes (0.0427%). Common emoji remained on the exact fast path.

### Exactness gates

- Fixed golden uint32 SHA256 values for multilingual, Indic, Romanized Indic, CJK, Arabic/Hebrew, Cyrillic/Greek, emoji, SQL, Python, JSON/chat, whitespace, controls, and mixed-script stress cases.
- Exact pre-token boundaries on whitespace/lookahead edge cases.
- Exact token IDs on 4,194,302 bytes stratified across the 1 GiB sample.
- Strongest check: all native 1 GiB modes emitted 215,432,465 tokens and the same ordered chunk digest as the completed reference run.
- Per-format production preflight compares native uint32 bytes to reference bytes before writing a shard.

## Alternative implementation investigation

- **Cython/Numba:** neither is installed. Accelerating the existing Python dictionaries and millions of tuple/list allocations would still require restructuring the trie and output representation; annotation alone would not address the hot path.
- **C++/pybind11:** `g++` is available but pybind11 is not. A compact C++ trie could use the same design, but it offers no semantic or expected performance advantage over the completed Rust/PyO3 path.
- **Hugging Face Tokenizers:** stock models are BPE, Unigram, WordPiece, and WordLevel. AGLM’s equal-cost shortest-path byte-trie model and exact tie-breaking are not representable as an ordinary BPE tokenizer JSON. The Rust crate exposes a `Model` trait, so a custom AGLM model is possible in a fork/integration, but exporting this vocabulary as stock BPE or Unigram is rejected because it changes IDs and cannot safely represent arbitrary base byte tokens.

## Production sample and 20 TB projection

The bounded native builder processed 117,353,781 raw bytes into 23,563,345 tokens and two verified shards at 12.286 MiB/s raw. Peak aggregate PSS was 3.62 GiB. It recorded one Unicode-table fallback document (1,048,439 bytes), no leakage, no retained exact duplicates, no temporary shards, and a passed final verification report. The mmap loader measured 40,013 samples/s and 81.946M tokens/s at sequence length 2,048.

Using the 1 GiB gate’s measured 4.984115805 raw bytes/token:

- 20 TB decimal raw ≈ 4.013 trillion tokens;
- uint32 storage ≈ 16.051 TB decimal;
- packed21 storage ≈ 10.533 TB decimal;
- old 1.066 MiB/s reference production pipeline projection: 207.0 days;
- measured 12.286 MiB/s complete native pipeline projection: 18.0 days;
- isolated 35.420 MiB/s eight-thread tokenizer ceiling: 6.23 days.

The 18-day value is a straight-line estimate, not authorization or a guarantee of sustained 20 TB throughput. The remaining gap between 35.4 MiB/s tokenizer-only and 12.3 MiB/s end-to-end is now outside the tokenizer hot loop: frequency accounting, main-thread document bookkeeping, shard writes/fsync, and verification are the next optimization targets if a sustained pipeline rate above 20 MiB/s is desired.

## Current gate state

- 100 MiB reference sample: passed.
- 1 GiB reference dataset gate: passed and paused.
- 1 GiB in-memory reference/native equivalence and scaling: passed.
- 117 MB native end-to-end sharding/loader gate: passed.
- Full corpus conversion: **NOT STARTED**.
