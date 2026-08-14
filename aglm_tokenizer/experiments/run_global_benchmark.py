"""
Master Global Multilingual Benchmark & Tokenizer Audit Suite.
Executes end-to-end evaluation covering all 17 mandatory requirements:
- Language coverage across Latin, Cyrillic, Arabic, Indic, Dravidian, East Asian, Southeast Asian, African, and others.
- Romanization stress testing (10,000+ examples).
- Code-switched & mixed-script sentence evaluation.
- Allocation strategies comparison (A through E).
- Public tokenizer comparison on the exact same benchmark.
- SHA-256 provenance registration.
"""

from typing import Dict, List, Any
import os
import sys
import time
import json
from tabulate import tabulate

from aglm_tokenizer.corpus.language_registry import LANGUAGES, ScriptFamily
from aglm_tokenizer.corpus.provenance import ProvenanceTracker
from aglm_tokenizer.corpus.multilingual_corpus import MultilingualCorpusManager
from aglm_tokenizer.corpus.code_switch_dataset import CodeSwitchDatasetGenerator
from aglm_tokenizer.allocation.strategies import AllocationStrategyType
from aglm_tokenizer.eval.benchmark_suite import MultilingualBenchmarkSuite
from aglm_tokenizer.eval.public_tokenizers import PublicTokenizerFactory, PublicTokenizerWrapper
from aglm_tokenizer.eval.romanization_audit import RomanizationAuditor
from aglm_tokenizer.experiments.run_vocab_sweep import VocabSweepRunner
from aglm_tokenizer.eval.metrics import MetricsCalculator, LanguageEvalResult, FairnessMetrics


class GlobalBenchmarkRunner:
    """Master orchestrator for the global multilingual tokenizer research and audit."""

    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.provenance_tracker = ProvenanceTracker()
        self.sweep_runner = VocabSweepRunner()

    def run_all(self) -> Dict[str, Any]:
        print("=" * 80)
        print("AGLM UNIVERSAL MULTILINGUAL TOKENIZER — RESEARCH & AUDIT SUITE")
        print("=" * 80)

        # ---------------------------------------------------------
        # 1. Register SHA-256 Data Provenance (Section 13)
        # ---------------------------------------------------------
        print("\n[STEP 1/6] Registering Cryptographic SHA-256 Data Provenance...")
        MultilingualCorpusManager.register_all_provenance(self.provenance_tracker)
        manifest_path = os.path.join(self.output_dir, "provenance_manifest.json")
        self.provenance_tracker.export_manifest(manifest_path)
        print(f"           Registered SHA-256 hashes for {len(self.provenance_tracker.records)} corpus slices.")
        print(f"           Provenance manifest saved to: {manifest_path}")

        # ---------------------------------------------------------
        # 2. Build AGLM Candidate Tokenizers (Section 4, 5, 6, 7)
        # ---------------------------------------------------------
        print("\n[STEP 2/6] Building AGLM Universal Multilingual Tokenizers (Fairness-Aware Strategy E)...")
        aglm_256k = self.sweep_runner.train_tokenizer(256000, AllocationStrategyType.E_FAIRNESS_AWARE_UTILITY)
        aglm_512k = self.sweep_runner.train_tokenizer(512000, AllocationStrategyType.E_FAIRNESS_AWARE_UTILITY)
        aglm_save_dir = os.path.join(self.output_dir, "aglm_256k_model")
        aglm_256k.save(aglm_save_dir)
        print(f"           AGLM Universal 256K saved to: {aglm_save_dir}")

        # ---------------------------------------------------------
        # 3. Load Public Tokenizers (Section 14)
        # ---------------------------------------------------------
        print("\n[STEP 3/6] Loading Public Multilingual Production Tokenizers...")
        public_tokenizers = PublicTokenizerFactory.load_all_available()
        print(f"           Loaded {len(public_tokenizers)} public tokenizers: {list(public_tokenizers.keys())}")

        all_tokenizers: Dict[str, Any] = {
            "AGLM Universal (256K) [Ours]": aglm_256k,
            "AGLM Universal (512K) [Ours]": aglm_512k,
            **public_tokenizers
        }

        # ---------------------------------------------------------
        # 4. Run Language-by-Language Benchmark Table & Fairness (Section 8, 9, 14)
        # ---------------------------------------------------------
        print("\n[STEP 4/6] Executing Global Held-Out Multilingual Benchmark Matrix across 50+ Languages...")
        bench = MultilingualBenchmarkSuite()
        eval_matrices: Dict[str, List[LanguageEvalResult]] = {}
        fairness_summaries: Dict[str, FairnessMetrics] = {}

        comparison_rows = []
        for name, tok in all_tokenizers.items():
            print(f"           Evaluating: {name}...")
            results, fairness = bench.evaluate_tokenizer(name, tok)
            eval_matrices[name] = results
            fairness_summaries[name] = fairness

            vocab_sz = tok.vocab_size if hasattr(tok, "vocab_size") else "N/A"
            comparison_rows.append([
                name,
                f"{vocab_sz:,}" if isinstance(vocab_sz, int) else str(vocab_sz),
                f"{fairness.macro_mean:.2f}",
                f"{fairness.median_bytes_per_token:.2f}",
                f"{fairness.worst_bytes_per_token:.2f}",
                f"{fairness.best_bytes_per_token:.2f}",
                f"{fairness.p10:.2f} / {fairness.p90:.2f}",
                f"{fairness.coefficient_of_variation:.3f}",
                f"{fairness.gini_index:.3f}",
                f"{fairness.max_to_min_ratio:.2f}x"
            ])

        comp_headers = ["Tokenizer", "Vocab", "Macro B/T", "Median B/T", "Worst B/T", "Best B/T", "P10 / P90", "CV", "Gini", "Max/Min Ratio"]
        cross_tokenizer_table = tabulate(comparison_rows, headers=comp_headers, tablefmt="github")

        # Save individual detailed tables for key models
        aglm_table = bench.format_markdown_table(eval_matrices["AGLM Universal (256K) [Ours]"], "AGLM Universal (256K)")
        with open(os.path.join(self.output_dir, "aglm_256k_language_table.md"), "w", encoding="utf-8") as f:
            f.write(aglm_table)

        # ---------------------------------------------------------
        # 5. Romanization Stress Test: 10,000+ Examples (Section 2, 15)
        # ---------------------------------------------------------
        print("\n[STEP 5/6] Executing Romanization & Spelling-Variant Stress Test (10,000+ Examples)...")
        auditor = RomanizationAuditor(target_sample_count=10000)
        roman_audit_results = {}
        roman_comp_rows = []

        for name, tok in all_tokenizers.items():
            print(f"           Auditing romanization on: {name}...")
            audit = auditor.run_audit(name, tok)
            roman_audit_results[name] = audit
            roman_comp_rows.append([
                name,
                f"{audit['overall_bytes_per_token']:.2f}",
                f"{audit['overall_tokens_per_word']:.2f}",
                f"{audit['mean_spelling_fragmentation_variance']:.3f}",
                "100% Lossless" if audit["is_100_percent_lossless"] else f"FAILED ({audit['lossless_failures']} errs)"
            ])

        roman_headers = ["Tokenizer", "Roman B/T", "Tokens/Word", "Spelling Frag Var", "Lossless Roundtrip"]
        roman_comp_table = tabulate(roman_comp_rows, headers=roman_headers, tablefmt="github")

        # ---------------------------------------------------------
        # 6. Code-Switching & Mixed-Script Test (Section 3)
        # ---------------------------------------------------------
        print("\n[STEP 6/6] Evaluating Code-Switched & Mixed-Script Pairs...")
        cs_corpus = CodeSwitchDatasetGenerator.get_combined_corpus()
        cs_rows = []
        for name, tok in all_tokenizers.items():
            total_b = sum(len(item["text"].encode("utf-8")) for item in cs_corpus)
            total_toks = sum(len(tok.encode(item["text"])) for item in cs_corpus)
            cs_bpt = total_b / total_toks if total_toks > 0 else 0.0
            cs_rows.append([name, f"{cs_bpt:.2f}", f"{total_toks:,}"])

        cs_table = tabulate(cs_rows, headers=["Tokenizer", "Code-Switch B/T", "Total Tokens"], tablefmt="github")

        # ---------------------------------------------------------
        # Vocab Size Sweep Report (Section 10) & 4 Representations
        # ---------------------------------------------------------
        sweep_output = self.sweep_runner.run_sweep()

        print("\n" + "=" * 80)
        print("RESEARCH RESULTS & BENCHMARK SUMMARY")
        print("=" * 80)
        print("\n### 1. Cross-Tokenizer Global Multilingual Comparison Matrix\n")
        print(cross_tokenizer_table)

        print("\n### 2. Romanization & Transliteration Stress Test (10,000+ Examples)\n")
        print(roman_comp_table)

        print("\n### 3. Code-Switched & Mixed-Script Performance\n")
        print(cs_table)

        print("\n### 4. Global Vocabulary Scaling Table (96K to 2M across Strategies A through E)\n")
        print(sweep_output["scaling_table"])

        print("\n### 5. Model Representation Footprint & Parameter Overhead (Reps A, B, C, D)\n")
        print(sweep_output["rep_table"])

        print("\n### 6. Two Independent Ranking Winners\n")
        print(f"**1. Tokenizer Compression Winner (Ignores Model Embedding Cost)**: `{sweep_output['compression_winner']}`")
        print(f"**2. End-to-End Architecture Winner (Incorporates Model Embedding/Head Cost)**: `{sweep_output['e2e_winner']}`")

        return {
            "cross_tokenizer_table": cross_tokenizer_table,
            "aglm_language_table": aglm_table,
            "roman_comp_table": roman_comp_table,
            "cs_table": cs_table,
            "scaling_table": sweep_output["scaling_table"],
            "rep_table": sweep_output["rep_table"],
            "compression_winner": sweep_output["compression_winner"],
            "e2e_winner": sweep_output["e2e_winner"],
            "eval_matrices": eval_matrices,
            "fairness_summaries": fairness_summaries,
            "roman_audit_results": roman_audit_results
        }
