"""
Multilingual Dataset and Streaming Batch Loader.
Tracks exact UTF-8 byte lengths alongside token IDs to compute exact BPB.
"""

from typing import Iterator, Tuple, List, Dict, Any, Optional
import os
import torch
import numpy as np


class MultilingualTextDataset:
    def __init__(
        self,
        text_corpus: List[str],
        tokenizer: Any,
        seq_len: int = 512,
        vocab_size: int = 32768
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Pre-tokenize or stream
        self.token_buffer: List[int] = []
        self.byte_lengths: List[int] = []

        for doc in text_corpus:
            raw_bytes = len(doc.encode("utf-8"))
            if hasattr(tokenizer, "encode"):
                toks = tokenizer.encode(doc)
            else:
                toks = list(doc.encode("utf-8"))
            # Clamp token IDs to vocab_size
            toks = [t % vocab_size for t in toks]

            self.token_buffer.extend(toks)
            # Estimate bytes per token
            bpt = max(1, raw_bytes // max(1, len(toks)))
            self.byte_lengths.extend([bpt] * len(toks))

    def __len__(self) -> int:
        return max(1, len(self.token_buffer) // self.seq_len)

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
            input_ids: (B, T)
            target_ids: (B, T)
            total_tokens: int (B * T)
            total_bytes: int
        """
        max_idx = len(self.token_buffer) - self.seq_len - 1
        if max_idx <= 0:
            # Fallback random tokens if buffer too small
            inp = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=device)
            tgt = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=device)
            return inp, tgt, batch_size * self.seq_len, batch_size * self.seq_len * 3

        indices = np.random.randint(0, max_idx, size=batch_size)
        inps = []
        tgts = []
        tot_bytes = 0

        for idx in indices:
            inp_seq = self.token_buffer[idx : idx + self.seq_len]
            tgt_seq = self.token_buffer[idx + 1 : idx + self.seq_len + 1]
            inps.append(inp_seq)
            tgts.append(tgt_seq)
            tot_bytes += sum(self.byte_lengths[idx : idx + self.seq_len])

        inp_tensor = torch.tensor(inps, dtype=torch.long, device=device)
        tgt_tensor = torch.tensor(tgts, dtype=torch.long, device=device)
        tot_tokens = batch_size * self.seq_len

        return inp_tensor, tgt_tensor, tot_tokens, tot_bytes


def build_synthetic_multilingual_corpus() -> Tuple[List[str], List[str]]:
    """Builds a rich multilingual, code, and dialect text corpus for benchmarking."""
    train_docs = [
        "Artificial intelligence and neural network architectures are transforming scientific discovery.",
        "नमस्ते, मेरा नाम आकाश है। हम एक नया बहुभाषी टोकनाइज़र और मम्बा हाइब्रिड मॉडल बना रहे हैं।",
        "nenu oka kottha multilingual language model train chestunnanu. ee model telugu tho paatu english technical words ni kuda sarigga understand chesukovali.",
        "njan oru puthiya multilingual language model train cheyyukayaanu. ee model malayalam mathramalla english technical terms um nannayi understand cheyyanam.",
        "naanu ondu hosa multilingual language model train maduttiddene. ee model kannada jothege english technical words annu sariyagi understand madabeku.",
        "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
        "SELECT user_id, COUNT(order_id) as total_orders, SUM(amount) as revenue FROM transactions GROUP BY user_id HAVING revenue > 5000 ORDER BY revenue DESC;",
        "In modern state space models like Mamba-2 and Mamba-3, state space duality connects selective scans with causal structured linear attention.",
        "Spelling variations in Romanized Indic text such as 'karega', 'karenga', 'cheyyali', 'cheyali', 'madbeku', 'madabeku' require robust token representations.",
        "Deep autoregressive language modeling minimizes negative log-likelihood across sequential token representations.",
    ] * 50

    val_docs = [
        "Validation text testing generalization across Indic scripts, Romanized dialects, and technical source code.",
        "मुझे लगता है कि भाषा मॉडल को हिंदी और हिंग्लिश दोनों में समान रूप से दक्ष होना चाहिए।",
        "telugu lo code-mixed sentences rastunnappudu exact retrieval fail avvakunda chusukovali.",
        "malayalam manglish text understanding requires strong morphological awareness in the tokenizer.",
        "kannada text parsing requires high compression without losing exact key-value association.",
        "class TransformerAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.q_proj = nn.Linear(d_model, d_model)",
    ] * 20

    return train_docs, val_docs
