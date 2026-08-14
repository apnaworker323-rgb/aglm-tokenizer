"""
AGLM Universal Multilingual Tokenizer Package.
Designed for broad global multilingual coverage across all language families, scripts,
romanized forms, code-switched text, and balanced vocabulary allocations.
"""

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
from aglm_tokenizer.core.bpe_engine import BPEEngine
from aglm_tokenizer.core.script_handlers import ScriptDetector

__version__ = "1.0.0"

__all__ = [
    "AGLMUniversalTokenizer",
    "BPEEngine",
    "ScriptDetector",
]
