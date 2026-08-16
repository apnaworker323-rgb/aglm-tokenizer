"""Optional exact Rust accelerator for the AGLM shortest-path byte-trie model."""

from __future__ import annotations

from typing import List

import numpy as np

from aglm_tokenizer.core.script_handlers import ScriptSegmenter
from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer

try:
    from aglm_tokenizer._aglm_native import NativeBpe
except ImportError:  # pragma: no cover - optional extension
    NativeBpe = None


class AGLMNativeAccelerator:
    """Keep reference regex boundaries while moving trie/DP loops into Rust."""

    def __init__(self, tokenizer: AGLMUniversalTokenizer):
        if NativeBpe is None:
            raise ImportError(
                "AGLM native extension is not built. Run native/aglm_native/build.sh."
            )
        self.reference = tokenizer
        self.engine = NativeBpe(tokenizer.engine.id_to_bytes)

    def encode(self, text: str) -> List[int]:
        return self.engine.encode_segments(ScriptSegmenter.pre_tokenize(text))

    def encode_u32_bytes(self, text: str) -> bytes:
        return self.engine.encode_segments_u32(ScriptSegmenter.pre_tokenize(text))

    def encode_fully_native(self, text: str) -> List[int]:
        """Rust regex segmentation plus Rust trie/DP; must pass equivalence gates."""
        return self.engine.encode_text(text)

    def encode_fast(self, text: str) -> List[int]:
        """Exact linear-time Rust regex plus Rust trie/DP."""
        if self.requires_reference_fallback(text):
            return self.reference.encode(text)
        return self.engine.encode_text_fast(text)

    def encode_fast_u32_bytes(self, text: str) -> bytes:
        """Exact production path returning raw native-endian uint32 bytes.

        The extension is currently built only for little-endian production hosts;
        the dataset builder independently checks and records shard endianness.
        """
        if self.requires_reference_fallback(text):
            return np.asarray(self.reference.encode(text), dtype="<u4").tobytes()
        return self.engine.encode_text_fast_u32(text)

    def encode_minimal(self, text: str) -> List[int]:
        """Whole-document exact minimum-token segmentation, with no regex boundaries."""
        return self.engine.encode_text_minimal(text)

    def encode_minimal_u32_bytes(self, text: str) -> bytes:
        """Whole-document minimum segmentation as canonical little-endian uint32."""
        return self.engine.encode_text_minimal_u32(text)

    def requires_reference_fallback(self, text: str) -> bool:
        """True when Python Unicode-17 and Rust Unicode-16 classes may differ."""
        return self.engine.requires_reference_fallback(text)
