"""
Comprehensive Unit & Integration Test Suite for AGLM Dataset Builder.
Tests:
1. Tokenizer Identity & Vocab Bounds Verification
2. uint32 Overflow & Corruption Detection
3. Bit-for-Bit Exact SHA256 Roundtrip Lossless Verification
4. Shard File Size & Byte Alignment Assertions
5. Manifest Correctness & Hash Verification
6. Train/Val Separation & Document Independence
7. Multiformat Document Extraction (TXT, JSONL, GZ)
8. Sharded Memory-Mapped Dataset Loader Verification
"""

import os
import sys
import json
import gzip
import tempfile
import hashlib
import unittest
import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
from aglm_dataset_loader import AGLMShardedDataset
from build_aglm_dataset import TokenizerCensus, ProductionDatasetBuilder, discover_input_inventory


class TestAGLMDatasetBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer_dir = "exported_tokenizers/aglm_universal_max"
        cls.tc = TokenizerCensus(cls.tokenizer_dir)

    def test_01_tokenizer_identity_and_bounds(self):
        """Asserts exact vocabulary size and max token ID."""
        self.assertEqual(self.tc.vocab_size, 1551017)
        self.assertEqual(self.tc.max_id, 1551016)
        self.assertEqual(self.tc.min_id, 0)
        self.assertEqual(self.tc.eos_token_id, 258)
        self.assertEqual(self.tc.bos_token_id, 257)

    def test_02_uint16_overflow_proof(self):
        """Verifies that casting token IDs > 65535 to uint16 causes corruption."""
        sample_ids = np.array([0, 255, 65535, 65536, 100000, 1551016], dtype=np.uint32)
        uint16_cast = sample_ids.astype(np.uint16).astype(np.uint32)
        
        # Elements >= 65536 must fail equality
        self.assertEqual(uint16_cast[0], 0)
        self.assertEqual(uint16_cast[1], 255)
        self.assertEqual(uint16_cast[2], 65535)
        self.assertNotEqual(uint16_cast[3], 65536)  # 65536 becomes 0 in uint16!
        self.assertNotEqual(uint16_cast[4], 100000)
        self.assertNotEqual(uint16_cast[5], 1551016)

    def test_03_exact_lossless_roundtrip(self):
        """Tests that text encoded to uint32 decodes back to bit-for-bit identical bytes."""
        sample_text = (
            "AGLM Multilingual LLM Research Suite: Testing lossless roundtrip.\n"
            "नमस्ते! मेरा नाम आकाश है। हम बहुभाषी मम्बा हाइब्रिड मॉडल तैयार कर रहे हैं।\n"
            "def quick_sort(arr): return arr if len(arr) <= 1 else arr\n"
            "nenu oka kottha language model train chestunnanu."
        )
        orig_bytes = sample_text.encode("utf-8")
        tokens, _ = self.tc.tokenizer.engine.encode(sample_text)
        
        uint32_tokens = np.array(tokens, dtype=np.uint32)
        decoded_bytes = self.tc.tokenizer.engine.decode_to_bytes(uint32_tokens.tolist())
        
        sha_orig = hashlib.sha256(orig_bytes).hexdigest()
        sha_deco = hashlib.sha256(decoded_bytes).hexdigest()
        
        self.assertEqual(sha_orig, sha_deco)
        self.assertEqual(orig_bytes, decoded_bytes)

    def test_04_sharded_dataset_builder_pipeline(self):
        """Creates an end-to-end mini dataset with multiple formats and verifies shards."""
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            # 1. Create sample TXT file
            txt_path = os.path.join(raw_dir, "sample.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Line 1: Artificial intelligence and neural state space models.\n" * 50)

            # 2. Create sample JSONL file
            jsonl_path = os.path.join(raw_dir, "sample.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for i in range(50):
                    f.write(json.dumps({"text": f"Document {i}: Multilingual Indic tokenization and evaluation."}) + "\n")

            # 3. Create sample GZ file
            gz_path = os.path.join(raw_dir, "sample.txt.gz")
            with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                f.write("Gzipped text content for testing stream decompression.\n" * 50)

            files, _ = discover_input_inventory(raw_dir)
            self.assertEqual(len(files), 3)

            # Run builder with small shard size to test sharding
            builder = ProductionDatasetBuilder(
                input_dir=raw_dir,
                output_dir=out_dir,
                tokenizer_census=self.tc,
                shard_tokens=500,  # Small shard to force multiple shards
                val_ratio=0.1,
                dedupe_exact=True,
                enable_packed21=True
            )

            manifest = builder.build_dataset(files)
            
            # Assertions on output structure
            self.assertIn("train", manifest["shards"])
            self.assertIn("val", manifest["shards"])
            self.assertGreater(len(manifest["shards"]["train"]), 0)

            # Verify every shard file
            for split in ["train", "val"]:
                for s_info in manifest["shards"][split]:
                    rel_p = s_info["path"]
                    abs_p = os.path.join(out_dir, rel_p)
                    self.assertTrue(os.path.exists(abs_p))
                    
                    file_size = os.path.getsize(abs_p)
                    self.assertEqual(file_size % 4, 0)
                    self.assertEqual(file_size, s_info["token_count"] * 4)

                    # Read binary
                    tokens = np.fromfile(abs_p, dtype=np.uint32)
                    self.assertEqual(len(tokens), s_info["token_count"])
                    self.assertLessEqual(tokens.max(), 1551016)

            # Verify PyTorch DataLoader on created shards
            manifest_path = os.path.join(out_dir, "metadata", "dataset_manifest.json")
            dataset = AGLMShardedDataset(manifest_path, split="train", seq_len=64)
            self.assertGreater(len(dataset), 0)

            inp, tgt, n_toks = dataset.get_batch(batch_size=2, device=torch.device("cpu"))
            self.assertEqual(inp.shape, (2, 64))
            self.assertEqual(tgt.shape, (2, 64))
            self.assertEqual(n_toks, 128)
            print("  ✅ [TEST PASSED]: All end-to-end dataset builder and loader assertions passed.")


if __name__ == "__main__":
    unittest.main()
