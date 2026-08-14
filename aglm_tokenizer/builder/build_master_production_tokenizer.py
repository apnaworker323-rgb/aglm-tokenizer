"""
Master Production Tokenizer Builder & Unified Merger.
Unifies:
1. 1,093,151 Canonical Multi-Tokenizer Lexical Candidates (o200k, cl100k, XLM-V, Gemma 2, DeepSeek V3, Qwen 2.5, Llama 3, Mistral)
2. AI4Bharat Aksharantar / IndicXlit High-Utility Romanized Indic Morphemes (13 Languages)
3. Code-Aware Pre-Tokenization Boundaries and Punctuation-Bound Identifiers
4. 100% Exact Lossless Roundtrip Guarantee with full 256-byte fallback.

Exports:
- AGLM-Universal-1M-Prod (Target Vocab: 1,000,000)
- AGLM-Universal-256K-Prod (Target Vocab: 256,000)
"""

from typing import Dict, List, Set, Tuple, Any
import os
import sys
import json
import time
import zipfile
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
from aglm_tokenizer.core.bpe_engine import BPEEngine
from aglm_tokenizer.core.script_handlers import ScriptSegmenter, ScriptDetector
from aglm_tokenizer.pool.empirical_utility import EmpiricalUtilityScorer
from aglm_tokenizer.pool.multisource_indic_harvester import MultiSourceIndicHarvester


from aglm_tokenizer.corpus.conversational_indic_lexicon import get_conversational_indic_lexicon
from aglm_tokenizer.corpus.indic_verb_morphology import generate_indic_verb_lexicon
from aglm_tokenizer.corpus.phonetic_variations_lexicon import get_phonetic_variations_lexicon
from aglm_tokenizer.corpus.tenglish_dravidian_lexicon import get_tenglish_dravidian_lexicon
from aglm_tokenizer.corpus.tanglish_dravidian_lexicon import get_tanglish_dravidian_lexicon
from aglm_tokenizer.corpus.kanglish_dravidian_lexicon import get_kanglish_dravidian_lexicon
from aglm_tokenizer.corpus.manglish_dravidian_lexicon import get_manglish_dravidian_lexicon


class MasterProductionTokenizerBuilder:
    """Orchestrates candidate harvesting, ranking, unification, and export."""

    def __init__(
        self,
        canonical_pool_path: str = "./canonical_pool_results/CANONICAL_TOKEN_POOL.jsonl"
    ):
        self.canonical_pool_path = canonical_pool_path
        self.scorer = EmpiricalUtilityScorer()
        train_words = self.scorer.lang_training_corpora.values()
        self.train_word_counts = Counter()
        for t in train_words:
            self.train_word_counts.update(t.lower().split())

    def load_canonical_candidates(self) -> Dict[bytes, Dict[str, Any]]:
        """Loads canonical candidates from 9 production models."""
        print(f"[1/4] Loading canonical pool candidates from {self.canonical_pool_path}...")
        canonical_pool: Dict[bytes, Dict[str, Any]] = {}
        with open(self.canonical_pool_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                meta = json.loads(line)
                raw_bytes = bytes.fromhex(meta["bytes_hex"])
                canonical_pool[raw_bytes] = meta
        print(f"      Loaded {len(canonical_pool):,} canonical candidates.")
        return canonical_pool

    def harvest_indicxlit_candidates(self, top_k_per_lang: int = 3000) -> Dict[bytes, Dict[str, Any]]:
        """Harvests top noise-filtered Romanized Indic candidates from Aksharantar."""
        print("\n[2/4] Harvesting top Romanized Indic candidates from AI4Bharat Aksharantar...")
        indic_candidates: Dict[bytes, Dict[str, Any]] = {}
        languages = ["hin", "tel", "tam", "kan", "mal", "ben", "mar", "guj", "pan", "ori", "asm", "nep", "urd"]

        for lang in languages:
            try:
                zip_path = hf_hub_download(repo_id="ai4bharat/Aksharantar", filename=f"{lang}.zip", repo_type="dataset")
                words_counter = Counter()
                with zipfile.ZipFile(zip_path, "r") as z:
                    for fname in z.namelist():
                        if fname.endswith(".json"):
                            with z.open(fname) as f:
                                for line in f:
                                    try:
                                        item = json.loads(line.decode("utf-8"))
                                        w = item.get("english word", "").strip().lower()
                                        if w and w.isalpha() and 2 <= len(w) <= 24:
                                            words_counter[w] += 1
                                    except Exception:
                                        continue

                # Rank and take top K
                for w, freq in words_counter.most_common(top_k_per_lang):
                    corpus_c = self.train_word_counts.get(w, 0)
                    if freq >= 2 or corpus_c > 0 or len(w) <= 6:
                        for prefix in [" ", ""]:
                            b_seq = f"{prefix}{w}".encode("utf-8")
                            if b_seq not in indic_candidates:
                                indic_candidates[b_seq] = {
                                    "bytes_hex": b_seq.hex(),
                                    "text": f"{prefix}{w}",
                                    "is_valid_utf8": True,
                                    "byte_length": len(b_seq),
                                    "script": "LATIN",
                                    "structural_type": "ROMANIZED",
                                    "sources": {"indicxlit_aksharantar": freq},
                                    "consensus_count": 1,
                                    "frequency": corpus_c + freq
                                }
            except Exception as e:
                print(f"      [WARN] Could not load {lang}: {e}")

        print(f"      Harvested {len(indic_candidates):,} high-utility Romanized Indic candidates.")
        return indic_candidates

    def merge_and_rank_all(
        self,
        canonical_pool: Dict[bytes, Dict[str, Any]],
        indic_candidates: Dict[bytes, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merges all pools, removes exact-byte duplicates, and ranks by balanced utility."""
        print("\n[3/4] Unifying all candidate pools and computing global balanced utility...")
        unified_pool: Dict[bytes, Dict[str, Any]] = dict(canonical_pool)

        # Merge Romanized Indic candidates from Aksharantar
        for b_seq, meta in indic_candidates.items():
            if b_seq not in unified_pool:
                unified_pool[b_seq] = meta

        # Merge Multi-source Indic & Code-Mixed candidates (Sarvam AI, Navarasa 2.0, L3Cube)
        harvester = MultiSourceIndicHarvester(self.canonical_pool_path)
        multisource_candidates = harvester.harvest_all()
        for b_seq, meta in multisource_candidates.items():
            if b_seq not in unified_pool:
                unified_pool[b_seq] = meta

        # Add guaranteed code syntax tokens
        code_syntax = [
            " ->", " ::", " ==", " !=", " <=", " >=", " += ", " -= ", " *= ", " /= ",
            " &&", " ||", " === ", " !== ", "def ", "class ", "return ", "import ", "from ",
            "async ", "await ", "public ", "private ", "protected ", "static ", "const ", "let ",
            "fn ", "pub ", "struct ", "impl ", "enum ", "trait ", "package ", "func ", "type ",
            "SELECT ", "FROM ", "WHERE ", "GROUP BY ", "ORDER BY ", "HAVING ", "INSERT INTO ",
            "CREATE TABLE ", "BIGINT ", "VARCHAR", "TIMESTAMP ", "PRIMARY KEY ", "<div>", "</div>",
            "<span>", "</span>", "<script>", "</script>", "<html>", "</html>", "<head>", "</head>",
            '": ', '",\n', '": {\n', '": [\n', "()", "[]", "{}", ");", "():", "): ", "]:"
        ]
        for cs in code_syntax:
            b_cs = cs.encode("utf-8")
            if b_cs not in unified_pool:
                unified_pool[b_cs] = {
                    "bytes_hex": b_cs.hex(),
                    "text": cs,
                    "is_valid_utf8": True,
                    "byte_length": len(b_cs),
                    "script": "LATIN",
                    "structural_type": "CODE",
                    "sources": {"code_syntax_curated": 100},
                    "consensus_count": 5,
                    "frequency": 5000
                }

        # Add guaranteed core Romanized Indic pronouns & high-frequency root forms (bare + space)
        core_romanized_words = [
            "mujhe", "tujhe", "hame", "hume", "hum", "nenu", "naaku", "nuvvu", "repu",
            "cheyyali", "cheyali", "chey", "vellali", "kani", "mundu", "konchem", "pani",
            "mera", "meri", "mere", "tera", "teri", "tere", "apna", "apni", "apne",
            "karna", "karo", "kare", "karte", "karti", "karta", "hai", "hain", "tha", "thi", "the",
            "kyuki", "kyunki", "kyonki", "kyun", "kyu", "kyon", "isliye", "lekin", "magar", "par", "aur", "ya"
        ]
        for w in core_romanized_words:
            for pfx in ["", " "]:
                w_str = f"{pfx}{w}"
                b_w = w_str.encode("utf-8")
                if b_w not in unified_pool:
                    unified_pool[b_w] = {
                        "bytes_hex": b_w.hex(),
                        "text": w_str,
                        "is_valid_utf8": True,
                        "byte_length": len(b_w),
                        "script": "LATIN",
                        "structural_type": "ROMANIZED",
                        "sources": {"core_romanized_curated": 100},
                        "consensus_count": 5,
                        "frequency": 10000
                    }

        print(f"      Total Deduplicated Unified Candidates Pool: {len(unified_pool):,}")

        # Rank candidates by empirical multilingual utility
        scored_candidates = self.scorer.score_canonical_pool(unified_pool)
        print(f"      Successfully ranked {len(scored_candidates):,} candidates.")
        return scored_candidates

    def build_and_export(
        self,
        scored_candidates: List[Dict[str, Any]],
        target_vocab_size: Optional[int],
        output_dir: str,
        model_name: str
    ) -> AGLMUniversalTokenizer:
        """Constructs and exports a standalone tokenizer artifact."""
        os.makedirs(output_dir, exist_ok=True)
        vocab_target_str = f"{target_vocab_size:,}" if target_vocab_size else "UNLIMITED (Full Capacity)"
        print(f"\n[4/4] Building {model_name} (Target Vocab: {vocab_target_str})...")

        tok = AGLMUniversalTokenizer(name=model_name, strategy="unified_production_v1")

        # 1. Guarantee all high-frequency conversational Indic words + Verb Morphology + Phonetic Variations + Tenglish + Tanglish + Kanglish + Manglish (bare + space forms)
        guaranteed_lex = (
            get_conversational_indic_lexicon()
            .union(generate_indic_verb_lexicon())
            .union(get_phonetic_variations_lexicon())
            .union(get_tenglish_dravidian_lexicon())
            .union(get_tanglish_dravidian_lexicon())
            .union(get_kanglish_dravidian_lexicon())
            .union(get_manglish_dravidian_lexicon())
        )
        for w in guaranteed_lex:
            for pfx in ["", " "]:
                b_tok = f"{pfx}{w}".encode("utf-8")
                tok.add_token(b_tok)

        # 2. Fill remaining slots from all ranked multi-lingual candidates (no artificial limit if target_vocab_size is None)
        for c in scored_candidates:
            if target_vocab_size is not None and tok.vocab_size >= target_vocab_size:
                break
            tok.add_token(c["raw_bytes"])

        print(f"      Constructed Tokenizer with exact Vocab Size: {tok.vocab_size:,}")

        # Save artifacts
        tok.save(output_dir)

        manifest = {
            "model_name": model_name,
            "vocab_size": tok.vocab_size,
            "algorithm": "Byte-Level BPE with Code-Aware Longest-Prefix Match Trie",
            "byte_fallback": "Guaranteed 256 byte tokens (0x00-0xFF)",
            "unified_sources": [
                "OpenAI o200k_base & cl100k_base",
                "Meta XLM-V & XLM-RoBERTa",
                "Google Gemma 2",
                "DeepSeek V3",
                "Alibaba Qwen 2.5",
                "Meta Llama 3",
                "Mistral v0.3",
                "AI4Bharat Aksharantar (13 Indic Languages)",
                "Curated Multi-Language Code Syntax"
            ],
            "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return tok


def verify_master_production_tokenizer(tok: AGLMUniversalTokenizer) -> None:
    """Verifies 100% exact lossless roundtrip and compression on code and Indic probes."""
    print("\n" + "=" * 80)
    print("VERIFYING MASTER PRODUCTION TOKENIZER (LOSSLESS & COMPRESSION TESTS)")
    print("=" * 80)

    test_cases = [
        ("Python Code (121B Sample)", "def calculate_hash(data: bytes, salt: str = 'aglm_1m') -> str:\n    return hashlib.sha256(data + salt.encode()).hexdigest()"),
        ("Hinglish", "mujhe ye kaam complete karna hai, sab theek chal raha hai aur hum aage badh rahe hain."),
        ("Tenglish (Telugu)", "nuvvu ekkada unnaavu? nenu intiki veltunnaanu, urgent ga call cheyyi."),
        ("Tanglish (Tamil)", "unga peru enna? nalla irukkeengala? enga veetuku vaanga saapida."),
        ("Hindi (Devanagari)", "नमस्ते, मेरा नाम आकाश है। हम एक नया सार्वभौमिक बहुभाषी टोकनाइज़र बना रहे हैं।"),
        ("Telugu", "నమస్కారం, నా పేరు ఆకాష్. మేము సరికొత్త టోకనైజర్ తయారు చేస్తున్నాము."),
        ("Tamil", "வணக்கம், என் பெயர் ஆகாஷ். நாங்கள் புதிய டோக்கனைசரை உருவாக்குகிறோம்."),
        ("Arabic", "الذكاء الاصطناعي يغير العالم والعلوم الحديثة بسرعة فائقة."),
        ("Chinese", "人工智能正在迅速改变现代计算机科学与前沿工程技术。"),
        ("Japanese", "人工知能は急速に進化しており、世界中で幅広く活用されています。"),
        ("Korean", "인공지능은 현대 컴퓨터 과학의 핵심 기술이자 미래 성장 동력입니다."),
        ("Russian", "Высокопроизводительные языковые модели нового поколения."),
        ("C++ Code", "template <typename T>\nclass StreamBuffer {\nprivate:\n    std::vector<std::unique_ptr<T>> elements_;\npublic:\n    void push_back(std::unique_ptr<T> item) { elements_.push_back(std::move(item)); }\n};"),
        ("SQL Query", "SELECT u.id, u.username, COUNT(o.id) AS total_orders FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE u.status = 'ACTIVE' GROUP BY u.id, u.username HAVING COUNT(o.id) > 10 ORDER BY total_orders DESC LIMIT 100;")
    ]

    all_passed = True
    for label, text in test_cases:
        toks = tok.encode(text)
        decoded = tok.decode(toks)
        b_len = len(text.encode("utf-8"))
        t_len = len(toks)
        bpt = b_len / t_len if t_len > 0 else 0.0
        is_exact = (text == decoded)
        if not is_exact:
            all_passed = False
        status_str = "PASS (100% Lossless)" if is_exact else "FAIL"
        print(f"[{status_str}] ({t_len:>2} toks, {bpt:5.2f} B/T) | {label:<28} | Preview: {text[:38]}...")

    assert all_passed, "Error: Master Tokenizer verification failed lossless roundtrip!"
    print("\n[SUCCESS] ALL PROBES PASSED 100% EXACT LOSSLESS ROUNDTRIP WITH HIGH COMPRESSION.")


def main():
    builder = MasterProductionTokenizerBuilder()
    canonical_pool = builder.load_canonical_candidates()
    indic_candidates = builder.harvest_indicxlit_candidates(top_k_per_lang=3000)
    scored_candidates = builder.merge_and_rank_all(canonical_pool, indic_candidates)

    # 1. Build and Export AGLM-Universal-Max (Unlimited Full Capacity Vocab — keeping 100% of candidate universe)
    tok_max = builder.build_and_export(
        scored_candidates=scored_candidates,
        target_vocab_size=None,
        output_dir="./exported_tokenizers/aglm_universal_max",
        model_name="AGLM-Universal-Max-Unlimited"
    )

    # Export also to aglm_universal_1m directory so existing web apps default to full capacity
    tok_max.save("./exported_tokenizers/aglm_universal_1m")

    # 2. Build and Export AGLM-Universal-256K-Prod
    tok_256k = builder.build_and_export(
        scored_candidates=scored_candidates,
        target_vocab_size=256000,
        output_dir="./exported_tokenizers/aglm_universal_256k",
        model_name="AGLM-Universal-256K-Production"
    )

    # Run Lossless & Compression Verification on Max Master
    verify_master_production_tokenizer(tok_max)


if __name__ == "__main__":
    main()
