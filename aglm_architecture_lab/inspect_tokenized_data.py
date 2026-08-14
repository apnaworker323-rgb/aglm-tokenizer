"""
Data Transformation & Tokenization Inspector for AGLM Architecture Lab.
Displays how multilingual sentences, Romanized Indic dialects, and code are transformed
into token IDs, bytes-per-token weights, and next-token training batches.
"""

from typing import List, Dict, Any
import os
import sys
import json
import gzip
import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aglm_architecture_lab.data.dataloader import build_synthetic_multilingual_corpus, MultilingualTextDataset


def inspect_data():
    print("=" * 100)
    print("AGLM TOKENIZATION & DATA TRANSFORMATION INSPECTOR")
    print("=" * 100)

    # 1. Sample Sentences across different languages & domains
    sample_corpus = [
        {
            "category": "Hindi (Devanagari)",
            "text": "नमस्ते, मेरा नाम आकाश है। हम एक नया बहुभाषी टोकनाइज़र और मम्बा हाइब्रिड मॉडल बना रहे हैं।"
        },
        {
            "category": "Hinglish (Romanized Hindi)",
            "text": "mujhe lagta hai ki artificial intelligence aane wale kuch saalon mein bahut tezi se badalne wali hai."
        },
        {
            "category": "Telugu (Romanized)",
            "text": "nenu oka kottha multilingual language model train chestunnanu. GPU memory saripokapothe batch size tagginchali."
        },
        {
            "category": "Malayalam (Romanized)",
            "text": "njan oru puthiya multilingual language model train cheyyukayaanu. tokenizer compression nannayi benchmark cheyyanam."
        },
        {
            "category": "Kannada (Romanized)",
            "text": "naanu ondu hosa multilingual language model train maduttiddene. training tumba slow aadare architecture optimize madabeku."
        },
        {
            "category": "Python Code",
            "text": "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    return quick_sort([x for x in arr if x < pivot]) + [pivot] + quick_sort([x for x in arr if x > pivot])"
        },
        {
            "category": "SQL Code",
            "text": "SELECT user_id, COUNT(order_id) as total_orders, SUM(amount) as revenue FROM transactions GROUP BY user_id;"
        }
    ]

    # Check for exported tokenizer vocab
    vocab_path = "exported_tokenizers/aglm_universal_256k/aglm_vocab.json.gz"
    vocab_map = {}
    id_to_token = {}
    if os.path.exists(vocab_path):
        try:
            with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if "tokens" in data and isinstance(data["tokens"], list):
                for item in data["tokens"]:
                    tid = item.get("id")
                    tstr = item.get("str")
                    if tid is not None and tstr is not None:
                        vocab_map[tstr] = tid
                        id_to_token[tid] = tstr
            print(f"[INFO] Loaded AGLM 256K Vocabulary ({len(vocab_map):,} tokens)")
        except Exception as e:
            print(f"[WARNING] Could not load vocab file: {e}")

    print("\n" + "-" * 100)
    print("PHASE 1: SENTENCE-BY-SENTENCE TOKENIZATION & BYTE BREAKDOWN")
    print("-" * 100)

    for item in sample_corpus:
        cat = item["category"]
        raw_text = item["text"]
        raw_bytes = raw_text.encode("utf-8")
        num_bytes = len(raw_bytes)

        # Tokenize using simple BPE/subword logic or byte encoding
        if vocab_map:
            # Simple greedy match with vocab
            tokens = []
            token_strs = []
            idx = 0
            while idx < len(raw_text):
                matched = False
                for length in range(min(40, len(raw_text) - idx), 0, -1):
                    sub = raw_text[idx : idx + length]
                    if sub in vocab_map:
                        tokens.append(vocab_map[sub])
                        token_strs.append(sub)
                        idx += length
                        matched = True
                        break
                if not matched:
                    # Fallback single byte/char
                    char_bytes = raw_text[idx].encode("utf-8")
                    for b in char_bytes:
                        tokens.append(b)
                        token_strs.append(chr(b) if 32 <= b < 127 else f"\\x{b:02x}")
                    idx += 1
        else:
            tokens = list(raw_bytes)
            token_strs = [chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in raw_bytes]

        num_tokens = len(tokens)
        bytes_per_tok = num_bytes / max(1, num_tokens)

        print(f"\n📁 Category: {cat}")
        print(f"📄 Original Text:\n   \"{raw_text}\"")
        print(f"📊 Stats: Raw Bytes = {num_bytes} bytes | Tokens = {num_tokens} | Compression = {bytes_per_tok:.2f} bytes/token")
        print(f"🔢 Transformed Token IDs (First 15): {tokens[:15]} ... (total {num_tokens})")
        print(f"🧩 Subword Chunk Representation: {token_strs[:10]} ...")

    print("\n" + "-" * 100)
    print("PHASE 2: TRAINING BATCH TRANSFORMATION (INPUT_IDS -> TARGET_IDS)")
    print("-" * 100)

    class DemoTokenizer:
        def encode(self, text: str) -> List[int]:
            return [b for b in text.encode("utf-8")]

    train_docs, _ = build_synthetic_multilingual_corpus()
    dataset = MultilingualTextDataset(train_docs, DemoTokenizer(), seq_len=32, vocab_size=32768)

    inp, tgt, total_toks, total_bytes = dataset.get_batch(batch_size=2, device=torch.device("cpu"))

    print(f"Batch Shape: Input IDs {inp.shape} | Target IDs {tgt.shape}")
    print(f"Total Tokens in Batch: {total_toks} | Total UTF-8 Bytes in Batch: {total_bytes}")
    print(f"\nSample Input Sequence [Batch Item 0]:\n{inp[0].tolist()}")
    print(f"\nSample Target Sequence [Next-Token Target for Item 0]:\n{tgt[0].tolist()}")
    print(f"\nOffset Alignment Check: Input[0, 1:] == Target[0, :-1] -> {torch.equal(inp[0, 1:], tgt[0, :-1])} (Strictly Causal Shift!)")
    print("=" * 100)


if __name__ == "__main__":
    inspect_data()
