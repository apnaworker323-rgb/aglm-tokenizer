# AGLM exact Rust tokenizer accelerator

This optional PyO3 extension accelerates the verified AGLM tokenizer without
changing its tokenization semantics. The Python reference remains the identity
oracle. The dataset builder will use this backend only when `--tokenizer-backend
native` is explicit and the per-format preflight produces byte-identical uint32
IDs.

Build on the current little-endian host:

```bash
bash native/aglm_native/build.sh
python3 -m pytest -q
```

The optimized path preserves the reference pre-token boundaries and its
minimum-token shortest-path byte-trie algorithm. It does not substitute greedy
BPE or ranked merges. It uses a linear-time Rust Unicode regex plus an explicit
repair for the reference pattern's negative-lookahead whitespace boundary,
compact trie arrays, a whole-segment terminal fast path, reusable DP buffers,
direct little-endian uint32 output, and releases the Python GIL so a single trie
can be shared across threads.

The installed Python `regex` oracle uses Unicode 17.0 tables while the pinned
Rust regex engine uses Unicode 16.0 tables. The extension contains the exhaustive
4,734-scalar symmetric difference for every boundary-relevant Unicode property
referenced by the pattern. (`Extended_Pictographic` changes are excluded because
the earlier punctuation alternative provably shadows that branch.) If a document contains any such scalar, the wrapper records the event
and sends the complete document through the Python reference. This is deliberate:
speed never takes precedence over exact token IDs.

Profile it with Linux `perf`:

```bash
perf record -g --call-graph dwarf -o benchmark_results/tokenizer_native.perf.data -- \
  python3 native/aglm_native/perf_target.py \
  --sample-file /path/to/utf8_sample.txt --iterations 20
perf report --stdio -i benchmark_results/tokenizer_native.perf.data
```

Never accept this extension solely because it is fast. The golden multilingual,
code, whitespace, and mixed-script cases in `tests/test_dataset_builder.py` and
the corpus-sample equivalence gate must both pass.
