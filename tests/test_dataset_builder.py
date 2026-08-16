"""Production invariants for the AGLM dataset builder and mmap loader."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aglm_dataset_loader import AGLMShardedDataset
from build_aglm_dataset import (
    EXPECTED_MAX_ID,
    EXPECTED_NORMAL_ID_COUNT,
    EXPECTED_REGEX_VERSION,
    EXPECTED_VOCAB_SHA256,
    EXPECTED_VOCAB_SIZE,
    ProductionDatasetBuilder,
    TokenizerCensus,
    discover_input_inventory,
    pack_21bit_vectorized,
    preflight_roundtrip,
    stream_documents_from_file,
    unpack_21bit_vectorized,
)

try:
    from aglm_tokenizer.native import AGLMNativeAccelerator, NativeBpe
except ImportError:  # pragma: no cover - source-only installs may not build Rust
    AGLMNativeAccelerator = None
    NativeBpe = None


class TestAGLMDatasetBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tc = TokenizerCensus(str(ROOT / "exported_tokenizers" / "aglm_universal_max"))
        cls.workspace = tempfile.TemporaryDirectory()
        cls.base = Path(cls.workspace.name)
        cls.raw = cls.base / "raw"
        cls.output = cls.base / "output"
        cls.reports = cls.base / "reports"
        (cls.raw / "nested").mkdir(parents=True)
        cls.txt_text = "  Leading whitespace is preserved.\nनमस्ते multilingual world.\nFinal line without newline"
        (cls.raw / "nested" / "sample.txt").write_text(cls.txt_text, encoding="utf-8")
        with (cls.raw / "records.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"text": "JSONL primary document", "language": "en", "unused": "metadata is not ingested"}) + "\n")
            handle.write(json.dumps({"prompt": "What is AGLM?", "response": "A multilingual model.", "domain": "chat"}) + "\n")
            handle.write(json.dumps({"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "नमस्ते"}]}) + "\n")
            handle.write(json.dumps({"text": "JSONL primary document"}) + "\n")  # exact duplicate
            handle.write(json.dumps({"id": 7, "metadata": "ambiguous"}) + "\n")
        with gzip.open(cls.raw / "compressed.txt.gz", "wt", encoding="utf-8", newline="") as handle:
            handle.write("gzip document αβγ\nsecond gzip line\n")
        with gzip.open(cls.raw / "compressed.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"question": "प्रश्न", "answer": "उत्तर"}) + "\n")
        with (cls.raw / "table.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "language", "score"])
            writer.writeheader()
            writer.writerow({"text": "CSV quoted, text", "language": "en", "score": 1})
        (cls.raw / "array.json").write_text(json.dumps([{"content": "array item one"}, {"body": "array item two"}]), encoding="utf-8")
        (cls.raw / "ignored.bin").write_bytes(b"\x00\x01\x02")
        (cls.raw / "fake.txt").write_bytes(b"\x00" * 1024)

        cls.files, cls.inventory = discover_input_inventory(str(cls.raw), str(cls.output), verbose=False)
        cls.preflight = preflight_roundtrip(cls.files, cls.tc, list(("text", "content", "body", "document", "response", "prompt", "question", "answer", "messages")), 64, 1 << 20)
        builder = ProductionDatasetBuilder(
            str(cls.raw), str(cls.output), cls.tc, shard_tokens=24, val_ratio=0.5,
            dedupe_exact=True, enable_packed21=True, workers=1,
            text_chunk_bytes=64, max_document_bytes=1 << 20,
            inventory_summary=cls.inventory, report_dir=str(cls.reports),
        )
        cls.manifest = builder.build_dataset(cls.files)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.workspace.cleanup()

    def test_01_tokenizer_identity(self) -> None:
        self.assertEqual(self.tc.name, "AGLM-Universal-Max-Unlimited")
        self.assertEqual(self.tc.vocab_size, EXPECTED_VOCAB_SIZE)
        self.assertEqual(self.tc.normal_id_count, EXPECTED_NORMAL_ID_COUNT)
        self.assertEqual(self.tc.addressable_id_count, EXPECTED_VOCAB_SIZE)
        self.assertEqual(self.tc.max_id, EXPECTED_MAX_ID)
        self.assertEqual(self.tc.model_sha256, EXPECTED_VOCAB_SHA256)
        self.assertEqual(self.tc.regex_version, EXPECTED_REGEX_VERSION)
        self.assertEqual(self.tc.eos_token_id, 258)

    def test_02_uint32_and_uint16_overflow(self) -> None:
        ids = np.array([0, 65_535, 65_536, 1_000_000, EXPECTED_MAX_ID], dtype="<u4")
        self.assertTrue(np.array_equal(ids, ids.astype("<u4")))
        wrapped = ids.astype(np.uint16).astype(np.uint32)
        self.assertFalse(np.array_equal(ids, wrapped))
        self.assertTrue(np.all(ids < EXPECTED_VOCAB_SIZE))

    def test_03_exact_roundtrip(self) -> None:
        text = "English हिन्दी తెలుగు 中文 code: def f(x): return x + 1\n"
        tokens = np.asarray(self.tc.tokenizer.encode(text), dtype="<u4")
        decoded = self.tc.tokenizer.decode_to_bytes(tokens.tolist())
        self.assertEqual(hashlib.sha256(decoded).digest(), hashlib.sha256(text.encode()).digest())
        self.assertEqual(decoded, text.encode())

    def test_04_inventory_is_recursive_and_fail_closed(self) -> None:
        paths = {row["rel_path"] for row in self.files}
        self.assertIn("nested/sample.txt", paths)
        self.assertNotIn("ignored.bin", paths)
        self.assertNotIn("fake.txt", paths)
        ignored = {row["rel_path"]: row["reason"] for row in self.inventory["ignored_files"]}
        self.assertIn("binary", ignored["fake.txt"])

    def test_05_txt_jsonl_gzip_csv_and_json_parsers(self) -> None:
        by_name = {row["filename"]: row for row in self.files}
        txt_docs = list(stream_documents_from_file(by_name["sample.txt"], {"text"}, text_chunk_bytes=1 << 20))
        self.assertEqual("".join(doc["text"] for doc in txt_docs), self.txt_text)
        jsonl_docs = list(stream_documents_from_file(by_name["records.jsonl"], set(("text", "prompt", "response", "messages"))))
        self.assertEqual(jsonl_docs[0]["text"], "JSONL primary document")
        self.assertIn("<|field:prompt|>", jsonl_docs[1]["text"])
        self.assertIn("<|message:user|>", jsonl_docs[2]["text"])
        self.assertIsNone(jsonl_docs[-1]["text"])
        gzip_docs = list(stream_documents_from_file(by_name["compressed.txt.gz"], {"text"}, text_chunk_bytes=1 << 20))
        self.assertEqual(gzip_docs[0]["text"], "gzip document αβγ\nsecond gzip line\n")
        csv_docs = list(stream_documents_from_file(by_name["table.csv"], {"text"}))
        self.assertEqual(csv_docs[0]["text"], "CSV quoted, text")
        json_docs = list(stream_documents_from_file(by_name["array.json"], {"content", "body"}))
        self.assertEqual([row["text"] for row in json_docs], ["array item one", "array item two"])

    def test_06_preflight_samples_every_present_format(self) -> None:
        self.assertFalse(self.preflight["failures"])
        for ext, result in self.preflight["formats"].items():
            self.assertEqual(result["status"], "passed", ext)

    def test_07_packed21_reversibility(self) -> None:
        rng = np.random.default_rng(7)
        ids = rng.integers(0, EXPECTED_VOCAB_SIZE, size=101, dtype=np.uint32)
        packed = pack_21bit_vectorized(ids)
        decoded = unpack_21bit_vectorized(packed, len(ids))
        self.assertTrue(np.array_equal(ids, decoded))
        self.assertEqual(len(packed), ((len(ids) + 7) // 8) * 21)

    def test_08_manifest_and_shards(self) -> None:
        self.assertEqual(self.manifest["numpy_dtype"], "<u4")
        self.assertEqual(self.manifest["document_boundary"]["eos_token_id"], 258)
        self.assertEqual(self.manifest["document_boundary"]["bos_policy"], "none")
        for split in ("train", "val"):
            self.assertTrue(self.manifest["shards"][split])
            for row in self.manifest["shards"][split]:
                path = self.output / row["path"]
                self.assertEqual(path.stat().st_size, row["token_count"] * 4)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
                ids = np.fromfile(path, dtype="<u4")
                self.assertLess(int(ids.max()), EXPECTED_VOCAB_SIZE)
                self.assertEqual(int(ids.min()), row["min_token_id"])
                packed = self.output / row["packed21"]["path"]
                self.assertTrue(packed.is_file())
        for name in ("dataset_manifest.json", "tokenizer_info.json", "source_files.jsonl", "shard_index.json", "statistics.json", "conversion_config.json"):
            self.assertTrue((self.output / "metadata" / name).is_file(), name)

    def test_09_exact_dedupe_and_split_separation(self) -> None:
        db = sqlite3.connect(self.output / "metadata" / "checkpoint.sqlite3")
        try:
            duplicate_groups = db.execute("SELECT COUNT(*) FROM (SELECT content_hash FROM documents GROUP BY content_hash HAVING COUNT(*)>1)").fetchone()[0]
            leakage = db.execute("SELECT COUNT(*) FROM (SELECT content_hash FROM documents GROUP BY content_hash HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
        finally:
            db.close()
        self.assertEqual(duplicate_groups, 0)
        self.assertEqual(leakage, 0)
        self.assertGreaterEqual(self.manifest["statistics"]["exact_duplicates"], 1)
        self.assertGreater(self.manifest["statistics"]["tokens_avoided"], 0)

    def test_10_resume_does_not_retokenize_verified_shards(self) -> None:
        before = [(row["path"], row["sha256"]) for split in ("train", "val") for row in self.manifest["shards"][split]]
        builder = ProductionDatasetBuilder(
            str(self.raw), str(self.output), self.tc, shard_tokens=24, val_ratio=0.5,
            dedupe_exact=True, enable_packed21=True, workers=1, resume=True,
            text_chunk_bytes=64, max_document_bytes=1 << 20,
            inventory_summary=self.inventory, report_dir=str(self.reports),
        )
        resumed = builder.build_dataset(self.files)
        after = [(row["path"], row["sha256"]) for split in ("train", "val") for row in resumed["shards"][split]]
        self.assertEqual(before, after)

    def test_11_loader_is_deterministic_and_stays_in_shard(self) -> None:
        manifest = self.output / "metadata" / "dataset_manifest.json"
        dataset_a = AGLMShardedDataset(manifest=str(manifest), split="train", seq_len=4, seed=123, max_open_shards=1)
        dataset_b = AGLMShardedDataset(manifest=str(manifest), split="train", seq_len=4, seed=123, max_open_shards=1)
        try:
            for index in range(min(10, len(dataset_a))):
                self.assertEqual(dataset_a.sample_location(index), dataset_b.sample_location(index))
                a_in, a_target = dataset_a[index]
                b_in, b_target = dataset_b[index]
                self.assertTrue(np.array_equal(a_in.numpy(), b_in.numpy()))
                self.assertTrue(np.array_equal(a_target.numpy(), b_target.numpy()))
                self.assertTrue(np.array_equal(a_in.numpy()[1:], a_target.numpy()[:-1]))
            batch_in, batch_target, count = dataset_a.get_batch(3)
            self.assertEqual(tuple(batch_in.shape), (3, 4))
            self.assertEqual(tuple(batch_target.shape), (3, 4))
            self.assertEqual(count, 12)
        finally:
            dataset_a.close()
            dataset_b.close()

    def test_12_random_document_sample_decodes(self) -> None:
        db = sqlite3.connect(self.output / "metadata" / "checkpoint.sqlite3")
        db.row_factory = sqlite3.Row
        try:
            rows = db.execute("SELECT * FROM documents ORDER BY doc_key LIMIT 5").fetchall()
            for row in rows:
                path = self.output / row["shard_path"]
                ids = np.memmap(path, dtype="<u4", mode="r", offset=int(row["token_offset"]) * 4, shape=(int(row["token_count"]),))
                self.assertEqual(int(ids[-1]), self.tc.eos_token_id)
                decoded = self.tc.tokenizer.decode_to_bytes(ids[:-1].tolist())
                self.assertEqual(hashlib.sha256(decoded).hexdigest(), row["content_hash"])
                del ids
        finally:
            db.close()

    @unittest.skipIf(NativeBpe is None, "optional Rust extension is not built")
    def test_13_native_accelerator_is_bit_exact(self) -> None:
        native = AGLMNativeAccelerator(self.tc.tokenizer)
        cases = [
            "ASCII and whitespace:\talpha  beta\r\nfinal\n",
            "हिन्दी नमस्ते। বাংলা ગુજરાતી ਪੰਜਾਬੀ ଓଡ଼ିଆ தமிழ் తెలుగు ಕನ್ನಡ മലയാളം",
            "Romanized Indic: mera naam Akash hai; nenu baagunnanu.",
            "中文分词。日本語。한국어 토크나이저.",
            "العربية، עברית, Ελληνικά, Русский.",
            "👩🏽‍💻🚀 family 👨‍👩‍👧‍👦 ©™✓️",
            "def f(xs: list[int]):\n    return [x * 2 for x in xs]  # λ\n",
            "SELECT a.id, COUNT(*) FROM t AS a WHERE a.x >= 10 GROUP BY a.id;",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\\nworld\"}]}",
            "a  day | a  123 | a  中文 | x\u00a0 - y | a\u2003\u2003word",
            "literal controls: \\x00 \\xff <|eos|> are ordinary text",
            "",
        ]
        # Deterministic mixed-script stress case exercises many adjacent regex
        # classes and whitespace-lookahead boundary repairs.
        alphabet = ["a", "Z", "9", " ", "\t", "\n", "ह", "ि", "中", "ع", "👩", "_", "-", "'", "("]
        cases.append("".join(alphabet[(index * 17 + index // 7) % len(alphabet)] for index in range(20_000)))
        golden_uint32_sha256 = [
            "b6310f86cbbbfbc06ecdaf25c91b37b496ef236fbc10460916cef6a5d77f1b06",
            "3ad5bc1410e95ff24e1a25600a5b28e3c55cf1a28620a660ae541e94dea69453",
            "a1359d41232db4a01483c23198df1b046e00e4b6ed4c4985411abd577137c33d",
            "14cac5aaf0589cf5bb0ecd4e2ce230de4c2097f671efc84a72d542dd540aa455",
            "a356355d22be090f0233863707437d13548193e336547e8bae8c0c129ab4a486",
            "ae5c782c425ecce8f128d58affe9bee2af7566d7ef9bd21eae1684059e8bf411",
            "5bca417cc56c5d32d1fb38663cd6d93e40ec04a2aa122445e2d8a116c2ba8679",
            "0db477cb525b9d5b72297befe29e477e703fe79496f666c211e0eb668efe9ffa",
            "267cea112b0da12b336c6b10d18042c726a8171019327d09eeee21fa54c21121",
            "d6d8ca99a36d855ec9eed07558402c307d39a5c5298b84cdcdd7e3097dadde7c",
            "d8d293ac78cacbbe6aa0a98b0c86eb929abe5b564a824726f5e5b8847d581177",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "19da78b9275cef384ad066a77d3b101137cd13dd70207c10313cebff00cbc8b8",
        ]
        for text, golden in zip(cases, golden_uint32_sha256):
            reference_segments = __import__(
                "aglm_tokenizer.core.script_handlers", fromlist=["ScriptSegmenter"]
            ).ScriptSegmenter.pre_tokenize(text)
            self.assertEqual(reference_segments, native.engine.pre_tokenize_fast_exact(text))
            reference = np.asarray(self.tc.tokenizer.encode(text), dtype="<u4").tobytes()
            accelerated = native.encode_fast_u32_bytes(text)
            self.assertEqual(hashlib.sha256(reference).hexdigest(), golden)
            self.assertEqual(reference, accelerated)
            if accelerated:
                ids = np.frombuffer(accelerated, dtype="<u4")
                self.assertGreaterEqual(int(ids.min()), 0)
                self.assertLess(int(ids.max()), EXPECTED_VOCAB_SIZE)
        risky = "Unicode-version guard: ʕ ★ \U00010940"
        self.assertEqual(native.engine.unicode_table_risk_count(), 4_734)
        self.assertTrue(native.requires_reference_fallback(risky))
        self.assertFalse(native.requires_reference_fallback("ordinary हिन्दी 中文 👩🏽‍💻"))
        expected = np.asarray(self.tc.tokenizer.encode(risky), dtype="<u4").tobytes()
        self.assertEqual(native.encode_fast_u32_bytes(risky), expected)
        controls = "actual controls: \x00\x01 and ÿ"
        expected = np.asarray(self.tc.tokenizer.encode(controls), dtype="<u4").tobytes()
        self.assertEqual(native.encode_fast_u32_bytes(controls), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
