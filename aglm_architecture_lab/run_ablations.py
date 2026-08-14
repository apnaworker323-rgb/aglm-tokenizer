"""
Comprehensive Ablation Study on Model E (AGLM Universal Hybrid).
Ablates:
1. Factorized Lexical Dimension (d_lexical = 64 vs 96 vs 128 vs Full Dense Table).
2. Routing Threshold tau (tau = 0.3 vs 0.5 vs 0.7 vs 100% Dense Attention).
3. Mamba Recurrence vs Pure Attention Refinement.
"""

from typing import Dict, Any, List
import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aglm_architecture_lab.models.config import ArchConfig, count_parameters
from aglm_architecture_lab.models.aglm_universal_hybrid import AGLMUniversalHybridLM
from aglm_architecture_lab.data.dataloader import MultilingualTextDataset, build_synthetic_multilingual_corpus
from aglm_architecture_lab.training.trainer import ArchitectureTrainer


def run_ablations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("AGLM UNIVERSAL HYBRID: SYSTEMATIC ABLATION STUDY")
    print("=" * 80)

    train_docs, val_docs = build_synthetic_multilingual_corpus()
    vocab_size = 32768
    seq_len = 256

    class ByteTokenizer:
        def encode(self, text: str) -> List[int]:
            return [b % vocab_size for b in text.encode("utf-8")]
        def decode(self, ids: List[int]) -> str:
            return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")

    tokenizer = ByteTokenizer()
    train_dataset = MultilingualTextDataset(train_docs, tokenizer, seq_len=seq_len, vocab_size=vocab_size)
    val_dataset = MultilingualTextDataset(val_docs, tokenizer, seq_len=seq_len, vocab_size=vocab_size)

    base_cfg = ArchConfig(
        name="AGLM_Ablation_Base",
        vocab_size=vocab_size,
        d_model=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=2,
        d_head=64,
        d_state=16,
        expand=1,
        seq_len=seq_len,
        factorize_embeddings=True,
        d_lexical=96,
        tau_threshold=0.5,
        target_capacity=0.5,
        extra={"n_mem_layers": 2, "n_ref_layers": 2}
    )

    ablation_configs = {
        "E0: Baseline AGLM Hybrid (d_lex=96, tau=0.5)": base_cfg,
        "E1: Lexical Dim 64 (d_lex=64)": base_cfg.copy_with(d_lexical=64),
        "E2: Lexical Dim 128 (d_lex=128)": base_cfg.copy_with(d_lexical=128),
        "E3: Dense Unfactorized Table (d_lex=384)": base_cfg.copy_with(factorize_embeddings=False),
        "E4: Sparse Routing (tau=0.7, ~30% Attention)": base_cfg.copy_with(tau_threshold=0.7, target_capacity=0.3),
        "E5: Dense Refinement (tau=0.0, 100% Attention)": base_cfg.copy_with(tau_threshold=0.0, target_capacity=1.0),
        "E6: Single Mamba Block (n_mem=1)": base_cfg.copy_with(extra={"n_mem_layers": 1, "n_ref_layers": 2}),
    }

    ablation_results = []

    for name, cfg in ablation_configs.items():
        if device.type == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\n---> Running Ablation: {name}")
        model = AGLMUniversalHybridLM(cfg)
        p_counts = count_parameters(model)

        trainer = ArchitectureTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            max_steps=100,
            batch_size=2,
            grad_accum_steps=2,
            lr=6e-4
        )
        train_res = trainer.train()

        record = {
            "Ablation Variation": name,
            "Total Params": f"{p_counts['total']:,}",
            "Embed Params": f"{p_counts['embedding']:,}",
            "Backbone Params": f"{p_counts['backbone']:,}",
            "Val Loss": round(train_res["val_loss"], 4),
            "Val BPB": round(train_res["val_bpb"], 4),
            "Train tok/s": int(train_res["train_tokens_per_sec"]),
            "Peak VRAM (MB)": round(train_res["peak_vram_mb"], 1),
        }
        print(f"     Val Loss: {train_res['val_loss']:.4f} | Val BPB: {train_res['val_bpb']:.4f} | VRAM: {train_res['peak_vram_mb']:.1f} MB")
        ablation_results.append(record)

        del model, trainer

    df_abl = pd.DataFrame(ablation_results)
    print("\n" + "=" * 90)
    print("AGLM UNIVERSAL HYBRID ABLATION RESULTS TABLE")
    print("=" * 90)
    print(df_abl.to_string(index=False))
    return df_abl


if __name__ == "__main__":
    run_ablations()
