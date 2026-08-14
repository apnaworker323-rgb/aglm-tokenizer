# Comprehensive Audit of Existing Architecture & Research History

**Date**: August 14, 2026  
**Auditor**: Senior LLM Architecture Researcher & Systems Engineer  
**Scope**: Full repository analysis across current project (`/aglm_project/tokenizer`), training datasets (`/aglm_project/data`), and historical architecture research (`final_version`, `architecture_search_v1`, `memonce_mamba3`, `embedding_ablation_v1`, `embedding_quant_research`).

---

## 1. Executive Summary & Forensic Timeline

Our project has undergone multiple rigorous research cycles spanning tokenizer optimization, embedding factorizations, recurrent state-space models, causal routing, and hardware-constrained quantization on RTX 3050 hardware (6 GB VRAM, Ampere sm_86).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RESEARCH TRAJECTORY MAP                                       │
├──────────────────────────────┬─────────────────────────────┬────────────────────────────────────┤
│ Phase 1: Architecture Search │ Phase 2: Input Compression  │ Phase 3: Tokenizer Universe        │
│ • Transformer GQA Baseline   │ • Standard 83.9M Embedding  │ • Universal Canonical Pool (1.55M) │
│ • Mamba-1/2/3, Griffin       │ • Byte-Balanced 8D+UTF8 MLP │ • 7 Dravidian/Indic Lexicons       │
│ • Hymba & Falcon-H1 Hybrids  │ • Early 250-step illusion   │ • 3-Way Split Screen Workbench     │
│ • Gated DeltaNet-2 & KDA     │ • True 2000-step widening   │ • 100% Lossless Roundtrip          │
│ • MemOnce-Mamba3 + RecAttn   │   gap (+4.91% BPB penalty)  │ • Zero-Drop Tokenizer Engine       │
└──────────────────────────────┴─────────────────────────────┴────────────────────────────────────┘
```

---

## 2. Inventory of Existing Repositories, Datasets & Checkpoints

### 2.1. Active Core Tokenizer Workspace (`/aglm_project/tokenizer`)
* **Core Tokenizer Engine** (`aglm_tokenizer/core/`):
  * `tokenizer.py`: `AGLMUniversalTokenizer` supporting fast trie lookup, special tokens, and auto `.json` / `.json.gz` loading.
  * `bpe_engine.py`: Trie-based byte-level longest-prefix BPE encoder with $O(T)$ forward scan.
  * `script_handlers.py`: Unicode script classifier (Devanagari, Telugu, Tamil, Kannada, Malayalam, Arabic, CJK, Latin, Cyrillic).
* **Corpus & Lexicon Modules** (`aglm_tokenizer/corpus/`):
  * `conversational_indic_lexicon.py`, `indic_verb_morphology.py`, `phonetic_variations_lexicon.py`
  * `tenglish_dravidian_lexicon.py`, `tanglish_dravidian_lexicon.py`, `kanglish_dravidian_lexicon.py`, `manglish_dravidian_lexicon.py`
* **Builder & Harvester** (`aglm_tokenizer/builder/`, `aglm_tokenizer/pool/`):
  * `build_master_production_tokenizer.py`: Builds `AGLM-Universal-Max` (1,551,017 tokens) and `AGLM-Universal-256K` (256,000 tokens) with multi-source ingestion (Sarvam-1, Navarasa 2.0, L3Cube-Pune, Aksharantar).
* **Evaluation & Benchmark Suite** (`aglm_tokenizer/eval/`):
  * `compare_aglm_vs_tiktoken_excel.py`: 1,248 test cases evaluated across AGLM, OpenAI `o200k_base`, and `cl100k_base`.
  * `AGLM_vs_tiktoken_1248_examples.xlsx` & `AGLM_VS_TIKTOKEN_1248_BENCHMARK_REPORT.md`.
* **Interactive Web Workbench** (`web_app/`):
  * Real-time 3-Way Split Screen (`app.py`, `app.js`, `style.css`, `index.html`) comparing AGLM 1.55M vs OpenAI GPT-4o vs Google Gemma 2.

### 2.2. Training Datasets (`/aglm_project/data/`)
* `fineweb_combined_train_95.txt` (4.85 GB) & `fineweb_combined_val_5.txt` (253 MB).
* `lmsys_train_95.txt` (2.05 GB) & `lmsys_val_5.txt` (103 MB).
* Total combined tokenized and raw text corpus: ~7.2 GB of clean multi-domain English, code, conversation, and multilingual text.

### 2.3. Historical Architecture Studies (`.Trash-1000/files/`)
* **`architecture_search_v1`**: Matched parameter study across 8 candidate backbones (`Transformer_GQA_Flash`, `Griffin_RGLRU_LocalAttention`, `Hymba_Parallel_Attention_SSM`, `FalconH1_Parallel_Attention_Mamba2`, `Mamba3`, `GatedDeltaNet2`, `MixtureOfRecursions`, `KimiLinear_KDA_Hybrid`).
* **`final_version` & `memonce_mamba3`**:
  * Implemented `MemOnceLM`: 2x Mamba-3 blocks over full sequence + 3x shared Transformer refinement blocks across 3 recursions with causal threshold routing.
  * Implemented `TRUE_T3_4_FULL` (resident 4-bit weight training + FP16 row norms).
* **`embedding_ablation_v1` & `embedding_quant_research`**:
  * Comprehensive input factorization ablations (8D, 16D, 32D, 64D, 128D, polynomial lifts, HyperLift, anchor codebooks, byte-balanced Kronecker tables).

---

## 3. Deep Dive: Previous Model Architectures

### 3.1. The `MemOnce-Mamba3 + RecAttn` Model Body

#### Architectural Design & Mathematical Formulation:
```
Input Tokens [B, T]
       │
       ▼
   Embedding (standard 640D or byte_balanced)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. Memory Pass: 2× Full-Sequence Mamba-3 Blocks        │
│    • NO ROUTING: Every position t updates state        │
│      h_t = A_t h_{t-1} + B_t x_t (state continuity)    │
│    • d_model = 640, d_state = 128, expand = 2          │
└────────────────────────────────────────────────────────┘
       │  [B, T, 640]
       ▼
┌────────────────────────────────────────────────────────┐
│ 2. Refinement Pass: 3 Shared Transformer Blocks × R=3  │
│    • Causal Threshold Router: g_t = σ(W_r x_t)         │
│    • Selection: Token t is routed iff g_t > τ (τ=0.5)  │
│    • Causal: No cross-token top-k competition          │
│    • Gather routed subset -> 3× GQA blocks -> Scatter  │
└────────────────────────────────────────────────────────┘
       │
       ▼
   Final RMSNorm(640) ──► Un-tied LM Head (131,072)
```

#### Why `MemOnce` was constructed:
1. **SSM State-Safety Constraint**: Linear recurrences ($h_t = A_t h_{t-1} + B_t x_t$) depend on exact sequential history. Routing or subsampling tokens *before* or *inside* an SSM destroys state continuity.
2. **Attention Gathering Tolerance**: Attention operates via pairwise dot products; gathering an arbitrary subset of tokens is mathematically valid as long as positional encodings (RoPE) index their true original positions ($pos_i$).
3. **Compute Amortization**: Running SSM over all $T$ positions costs $O(T)$ linearly, while running expensive Attention only over a subset $\kappa \cdot T$ ($0.5 \times$) across $R=3$ shared recursions reduces FLOPs.

---

## 4. Analysis of Previous Routing Experiments

### 4.1. Top-k Routing Failure (Non-Causal Future Leakage)
* In early prototypes, `g.topk(k, dim=1)` was evaluated with $k = \text{round}(T \cdot \text{capacity})$.
* **Critical Flaw Discovered**: In top-k routing, whether token $t$ is selected depends on whether its router score $g_t$ exceeds the $k$-th highest score across the *entire sequence* $\{g_1, \dots, g_T\}$. Tokens at step $t=10$ competed against tokens at step $t=500$. Future high-entropy tokens suppressed early tokens.
* **Empirical Audit**: 9 out of 200 causal perturbation tests failed due to future-token leakage.

### 4.2. Threshold-Based Causal Routing
* **Solution**: `_route(g)` switched to strict per-token thresholding:
  $$\text{selected}_t = \mathbb{I}(g_t > \tau) \quad \text{where } g_t = \sigma(W_{\text{router}} x_t)$$
* Because $g_t$ depends solely on $x_t$ (which depends only on $x_{\le t}$), the routing decision is **100% causal by construction**.
* Auxiliary loss added to prevent gate collapse:
  $$\mathcal{L}_{\text{aux}} = \frac{1}{R} \sum_{r=1}^R (\bar{g}_r - \text{capacity})^2$$

---

## 5. Analysis of Input Embedding & Factorization Experiments

### 5.1. The 250-Step vs 2000-Step Convergence Reality
In `final_version/docs/RESULTS.md`, an essential empirical finding was recorded:

| Training Step | Standard Embedding (83.9M params) BPB | Byte-Balanced 8D+UTF8 (2.13M params) BPB | Difference ($\Delta$ BPB) |
|:---:|:---:|:---:|:---:|
| **250** | 2.53790 | 2.52014 | **-0.70% (Byte-Balanced Won Early)** |
| **500** | 2.23694 | 2.29421 | **+2.56% (Standard Overtook)** |
| **1000** | 2.02694 | 2.10946 | **+4.07% (Standard Widening)** |
| **2000** | **1.86546** | **1.95699** | **+4.91% (Standard Dominated)** |

### 5.2. Root Cause Analysis:
1. **Cold-Start Illusion**: At 250 steps (700K tokens / 131K vocab), each word is seen ~5 times. The standard 83.9M embedding table was undertrained, giving the structured byte-prior a temporary head start.
2. **Information Bottleneck**: At 2000+ steps, compressing token semantics into 8 latent dimensions created a hard rank bottleneck that prevented nuanced distinction between related concepts.
3. **Conclusion for New Architecture**: Extreme embedding compression (<16 dims) damages long-term perplexity. Embedding factorization must maintain sufficient dimensionality ($d_{\text{lexical}} \ge 128$) or employ tied representations.

---

## 6. Current Hardware & Systems Constraints

* **Compute Platform**: NVIDIA GeForce RTX 3050 Laptop/Desktop GPU
  * VRAM: **5.67 GiB (6,085 MB)**
  * Compute Capability: **sm_86 (Ampere)**
  * Host Memory: **32 GB RAM**
  * Host CPU: **8 vCPUs (Intel Core i7-6700 @ 3.40GHz)**
* **Precision & Execution Policy**:
  * BF16 autocast with PyTorch native SDPA (`FLASH_ATTENTION` / `EFFICIENT_ATTENTION`).
  * Micro-batching: $B_{\text{micro}} = 2$, gradient accumulation steps $= 4$ ($B_{\text{eff}} = 8$), $T_{\text{seq}} = 512$.
  * Peak VRAM budget ceiling: $\le 4.5 \text{ GiB}$ to guarantee zero CUDA OOMs.

---

## 7. Baseline Architecture Specifications (Current Reference Point)

| Component | Standard Transformer GQA Baseline | MemOnce-Mamba3 + RecAttn Baseline |
|:---|:---|:---|
| **Vocab Size ($V$)** | 131,072 (Benchmark baseline) / 256,000 / 1,551,017 | 131,072 / 256,000 |
| **Model Dimension ($d_{\text{model}}$)** | 640 | 640 |
| **Layers ($N$)** | 12 blocks | 2 Mamba-3 + 3 Refinement $\times$ 3 recursions |
| **Attention Heads ($H / H_{\text{kv}}$)** | 8 query heads / 2 KV heads (GQA 4:1) | 8 query heads / 2 KV heads |
| **Head Dimension ($d_{\text{head}}$)** | 80 | 80 |
| **FFN Expansion** | SwiGLU ($d_{\text{ffn}} = 1,728$, $8/3 \cdot d$) | SwiGLU ($d_{\text{ffn}} = 1,728$) |
| **Normalization** | Pre-RMSNorm ($\epsilon = 10^{-5}$) | Pre-RMSNorm ($\epsilon = 10^{-5}$) |
| **Positional Encoding** | RoPE ($\text{base} = 10,000$) | RoPE with gathered index support |
| **Backbone Parameters** | ~25.04M params | ~25.04M params |
| **Input / Head Parameters** | 83.88M (at $V=131\text{k}$) | 83.88M (at $V=131\text{k}$) |

---

## 8. Verification & Audit Sign-Off

* [x] Entire repository structure audited.
* [x] Previous Mamba, Griffin, Hymba, Falcon-H1, and MemOnce experiments reviewed and preserved.
* [x] Causal routing constraints and past failure modes documented.
* [x] Hardware constraints (RTX 3050 6GB) established.
* [x] No files overwritten or destroyed.
