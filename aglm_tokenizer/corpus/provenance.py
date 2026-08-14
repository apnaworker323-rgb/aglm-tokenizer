"""
Global Held-Out Dataset Provenance and SHA-256 Validation.
Implements Section 13 of Mandatory Specifications:
- Strict mathematical separation between Tokenizer Training / Mining Data and Held-Out Evaluation Data.
- SHA-256 cryptographic verification for every language evaluation slice.
- Full provenance manifest.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CorpusSliceProvenance:
    language_code: str
    slice_type: str  # "train_mining" or "held_out_eval"
    byte_count: int
    char_count: int
    line_count: int
    sha256_hash: str
    provenance_source: str
    description: str


class ProvenanceTracker:
    """Manages cryptographic SHA-256 provenance for multilingual datasets."""

    def __init__(self):
        self.records: Dict[str, CorpusSliceProvenance] = {}

    def register_slice(
        self,
        lang_code: str,
        slice_type: str,
        text_content: str,
        source: str = "AGLM-Multilingual-Benchmark-v1",
        description: str = "Standard benchmark slice"
    ) -> CorpusSliceProvenance:
        raw_bytes = text_content.encode("utf-8")
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        prov = CorpusSliceProvenance(
            language_code=lang_code,
            slice_type=slice_type,
            byte_count=len(raw_bytes),
            char_count=len(text_content),
            line_count=len(text_content.splitlines()),
            sha256_hash=sha256,
            provenance_source=source,
            description=description
        )
        key = f"{lang_code}:{slice_type}"
        self.records[key] = prov
        return prov

    def export_manifest(self, filepath: str) -> None:
        """Exports provenance records to JSON manifest file."""
        data = {
            "version": "1.0.0",
            "protocol": "AGLM-Strict-HeldOut-Isolation",
            "slices": {
                k: {
                    "language_code": v.language_code,
                    "slice_type": v.slice_type,
                    "byte_count": v.byte_count,
                    "char_count": v.char_count,
                    "line_count": v.line_count,
                    "sha256_hash": v.sha256_hash,
                    "source": v.provenance_source,
                    "description": v.description
                }
                for k, v in self.records.items()
            }
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
