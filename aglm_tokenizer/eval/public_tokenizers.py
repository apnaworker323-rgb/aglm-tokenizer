"""
Public Multilingual Production Tokenizer Wrapper Classes.
Guarantees 100% authentic, isolated, independent execution for each model.
No lambda closure captures.
"""

from typing import Dict, List, Any, Optional
import time
import tiktoken
from transformers import AutoTokenizer


class PublicTokenizerWrapper:
    """Base wrapper guaranteeing isolated tokenizer instance dispatch."""

    def __init__(self, name: str, vocab_size: int):
        self.name = name
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError

    def decode_piece(self, token_id: int) -> str:
        return self.decode([token_id])

    def evaluate_text(self, text: str) -> Dict[str, Any]:
        raw_bytes = text.encode("utf-8")
        num_bytes = len(raw_bytes)
        words = text.split()
        num_words = max(1, len(words))

        t0 = time.perf_counter()
        tokens = self.encode(text)
        encode_time = time.perf_counter() - t0

        num_tokens = len(tokens)
        bytes_per_token = (num_bytes / num_tokens) if num_tokens > 0 else 0.0
        tokens_per_word = num_tokens / num_words
        throughput_mb_s = (num_bytes / (1024 * 1024)) / encode_time if encode_time > 0 else 0.0

        reconstructed = self.decode(tokens)
        is_lossless = (reconstructed == text)

        return {
            "tokens": tokens,
            "num_tokens": num_tokens,
            "num_bytes": num_bytes,
            "num_words": num_words,
            "bytes_per_token": bytes_per_token,
            "tokens_per_word": tokens_per_word,
            "encode_time_sec": encode_time,
            "throughput_mb_s": throughput_mb_s,
            "is_lossless": is_lossless
        }


class TiktokenWrapper(PublicTokenizerWrapper):
    """Direct wrapper around tiktoken Encoding instance."""

    def __init__(self, name: str, encoding_name: str):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        super().__init__(name=name, vocab_size=self.enc.n_vocab)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    def decode_piece(self, token_id: int) -> str:
        b = self.enc.decode_single_token_bytes(token_id)
        return b.decode("utf-8", errors="replace")


class HFAutoTokenizerWrapper(PublicTokenizerWrapper):
    """Direct wrapper around HuggingFace AutoTokenizer instance."""

    def __init__(self, name: str, model_id: str):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        v_size = self.tok.vocab_size if hasattr(self.tok, "vocab_size") else len(self.tok)
        super().__init__(name=name, vocab_size=v_size)

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        return self.tok.decode(tokens)

    def decode_piece(self, token_id: int) -> str:
        return self.tok.decode([token_id])


class PublicTokenizerFactory:
    """Instantiates verified independent tokenizer wrappers."""

    @classmethod
    def load_all_available(cls) -> Dict[str, PublicTokenizerWrapper]:
        tokenizers: Dict[str, PublicTokenizerWrapper] = {}

        # 1. OpenAI o200k_base
        try:
            tokenizers["OpenAI o200k_base"] = TiktokenWrapper("OpenAI o200k_base", "o200k_base")
        except Exception as e:
            print(f"[NOT BENCHMARKED] OpenAI o200k_base: {e}")

        # 2. OpenAI cl100k_base
        try:
            tokenizers["OpenAI cl100k_base"] = TiktokenWrapper("OpenAI cl100k_base", "cl100k_base")
        except Exception as e:
            print(f"[NOT BENCHMARKED] OpenAI cl100k_base: {e}")

        # 3. Qwen 2.5
        try:
            tokenizers["Qwen 2.5"] = HFAutoTokenizerWrapper("Qwen 2.5", "Qwen/Qwen2.5-7B")
        except Exception as e:
            print(f"[NOT BENCHMARKED] Qwen 2.5: {e}")

        # 4. Gemma 2
        try:
            tokenizers["Gemma 2"] = HFAutoTokenizerWrapper("Gemma 2", "unsloth/gemma-2-9b")
        except Exception as e:
            print(f"[NOT BENCHMARKED] Gemma 2: {e}")

        # 5. DeepSeek V3
        try:
            tokenizers["DeepSeek V3"] = HFAutoTokenizerWrapper("DeepSeek V3", "deepseek-ai/DeepSeek-V3")
        except Exception as e:
            print(f"[NOT BENCHMARKED] DeepSeek V3: {e}")

        # 6. Llama 3
        try:
            tokenizers["Llama 3"] = HFAutoTokenizerWrapper("Llama 3", "NousResearch/Meta-Llama-3-8B")
        except Exception as e:
            print(f"[NOT BENCHMARKED] Llama 3: {e}")

        # 7. Mistral v0.3
        try:
            tokenizers["Mistral v0.3"] = HFAutoTokenizerWrapper("Mistral v0.3", "mistralai/Mistral-7B-v0.3")
        except Exception as e:
            print(f"[NOT BENCHMARKED] Mistral v0.3: {e}")

        # 8. XLM-RoBERTa
        try:
            tokenizers["XLM-RoBERTa"] = HFAutoTokenizerWrapper("XLM-RoBERTa", "xlm-roberta-base")
        except Exception as e:
            print(f"[NOT BENCHMARKED] XLM-RoBERTa: {e}")

        # 9. XLM-V
        try:
            tokenizers["XLM-V"] = HFAutoTokenizerWrapper("XLM-V", "facebook/xlm-v-base")
        except Exception as e:
            print(f"[NOT BENCHMARKED] XLM-V: {e}")

        return tokenizers
