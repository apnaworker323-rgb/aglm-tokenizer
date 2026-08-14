"""
Script-Specific Handlers and Boundary Segmenters.
Implements Section 7 of Mandatory Specifications:
- Whitespace-delimited scripts
- Unspaced scripts (Chinese, Japanese, Thai, Myanmar, Khmer, Lao)
- Combining marks / diacritics preservation without unrecoverable normalization
- Arabic shaping / Harakat handling
- Indic conjuncts (Virama / Halant / Nukta / Matra preservation)
- Korean Hangul decomposition and syllable boundary respect
- Emoji / ZWJ sequence preservation
- 100% Exact Lossless Reconstruction Guarantee
"""

from enum import Enum
import unicodedata
import regex as re
from typing import List, Tuple, Optional


class ScriptType(str, Enum):
    LATIN = "LATIN"
    CYRILLIC = "CYRILLIC"
    ARABIC = "ARABIC"
    DEVANAGARI = "DEVANAGARI"
    BENGALI = "BENGALI"
    GUJARATI = "GUJARATI"
    GURMUKHI = "GURMUKHI"
    ODIA = "ODIA"
    TAMIL = "TAMIL"
    TELUGU = "TELUGU"
    KANNADA = "KANNADA"
    MALAYALAM = "MALAYALAM"
    CJK_HAN = "CJK_HAN"
    JAPANESE_KANA = "JAPANESE_KANA"
    HANGUL = "HANGUL"
    THAI = "THAI"
    MYANMAR = "MYANMAR"
    KHMER = "KHMER"
    LAO = "LAO"
    HEBREW = "HEBREW"
    GREEK = "GREEK"
    ARMENIAN = "ARMENIAN"
    GEORGIAN = "GEORGIAN"
    ETHIOPIC = "ETHIOPIC"
    EMOJI_SYMBOL = "EMOJI_SYMBOL"
    PUNCTUATION_SPACE = "PUNCTUATION_SPACE"
    NUMERIC = "NUMERIC"
    UNKNOWN = "UNKNOWN"


class ScriptDetector:
    """Detects the primary script of characters or text chunks."""

    @staticmethod
    def detect_char_script(ch: str) -> ScriptType:
        cp = ord(ch)

        # ASCII / Latin
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
            return ScriptType.LATIN

        # Numbers
        if 0x0030 <= cp <= 0x0039 or unicodedata.category(ch) == 'Nd':
            return ScriptType.NUMERIC

        # Punctuation / Space
        cat = unicodedata.category(ch)
        if cat.startswith('P') or cat.startswith('Z') or cat.startswith('C'):
            return ScriptType.PUNCTUATION_SPACE

        # Cyrillic
        if (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F):
            return ScriptType.CYRILLIC

        # Arabic
        if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0x08A0 <= cp <= 0x08FF):
            return ScriptType.ARABIC

        # Indic Scripts
        if 0x0900 <= cp <= 0x097F:
            return ScriptType.DEVANAGARI
        if 0x0980 <= cp <= 0x09FF:
            return ScriptType.BENGALI
        if 0x0A00 <= cp <= 0x0A7F:
            return ScriptType.GURMUKHI
        if 0x0A80 <= cp <= 0x0AFF:
            return ScriptType.GUJARATI
        if 0x0B00 <= cp <= 0x0B7F:
            return ScriptType.ODIA
        if 0x0B80 <= cp <= 0x0BFF:
            return ScriptType.TAMIL
        if 0x0C00 <= cp <= 0x0C7F:
            return ScriptType.TELUGU
        if 0x0C80 <= cp <= 0x0CFF:
            return ScriptType.KANNADA
        if 0x0D00 <= cp <= 0x0D7F:
            return ScriptType.MALAYALAM

        # East Asian
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF):
            return ScriptType.CJK_HAN
        if (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
            return ScriptType.JAPANESE_KANA
        if (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F):
            return ScriptType.HANGUL

        # Southeast Asian
        if 0x0E00 <= cp <= 0x0E7F:
            return ScriptType.THAI
        if 0x1000 <= cp <= 0x109F:
            return ScriptType.MYANMAR
        if 0x1780 <= cp <= 0x17FF:
            return ScriptType.KHMER
        if 0x0E80 <= cp <= 0x0EFF:
            return ScriptType.LAO

        # Other Scripts
        if 0x0590 <= cp <= 0x05FF:
            return ScriptType.HEBREW
        if 0x0370 <= cp <= 0x03FF:
            return ScriptType.GREEK
        if 0x0530 <= cp <= 0x058F:
            return ScriptType.ARMENIAN
        if 0x10A0 <= cp <= 0x10FF:
            return ScriptType.GEORGIAN
        if 0x1200 <= cp <= 0x137F:
            return ScriptType.ETHIOPIC

        if cat.startswith('S') or (0x1F300 <= cp <= 0x1FAFF):
            return ScriptType.EMOJI_SYMBOL

        return ScriptType.UNKNOWN

    @classmethod
    def detect_text_script(cls, text: str) -> ScriptType:
        counts: dict = {}
        for ch in text:
            st = cls.detect_char_script(ch)
            if st not in (ScriptType.PUNCTUATION_SPACE, ScriptType.NUMERIC, ScriptType.UNKNOWN):
                counts[st] = counts.get(st, 0) + 1
        if not counts:
            return ScriptType.LATIN
        return max(counts, key=counts.get)


class ScriptSegmenter:
    """
    Advanced pre-tokenization regex pattern engine designed for global multilingual coverage.
    Preserves:
    - Indic conjuncts (consonant + virama + consonant + matra + anusvara/visarga)
    - Arabic words with optional harakat (tashkeel)
    - Unspaced CJK characters as individual or dual characters
    - Thai / Myanmar / Khmer clusters
    - Emoji with ZWJ and skin tone modifiers
    - Numbers, URLs, contraction patterns for Latin languages
    - Contiguous whitespace / newlines
    """

    # Comprehensive multilingual split pattern
    # 1. Contractions and Latin words
    # 2. Indic syllable clusters (Devanagari, Bengali, Tamil, Telugu, etc. with virama/matra)
    # 3. Arabic words with diacritics
    # 4. Cyrillic, Greek, Armenian, Georgian, Hebrew words
    # 5. CJK Han characters (unspaced scripts - 1-2 char chunks)
    # 6. Hangul syllables
    # 7. Thai / Lao / Myanmar / Khmer clusters
    # 8. Emojis with ZWJ and modifiers
    # 9. Numbers (with separators)
    # 10. Whitespace runs
    # 11. Punctuation runs
    # 12. Fallback single characters / byte boundaries

    MULTILINGUAL_REGEX_PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|"""  # Latin contractions
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lo}]\p{M}*[\p{Ll}\p{Lo}]\p{M}*(?:'(?i:[sdmt]|ll|ve|re))?|"""  # Capitalized words with marks
        r"""[^\r\n\p{L}\p{N}]?[\p{Ll}\p{Lo}]\p{M}*(?:'(?i:[sdmt]|ll|ve|re))?|"""  # Lowercase words with marks
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lo}]\p{M}*(?:'(?i:[sdmt]|ll|ve|re))?|"""  # Upper/other letter words
        r"""\p{N}{1,4}|"""  # Numbers up to 4 digits
        r""" ?[^\s\p{L}\p{N}]+|"""  # Punctuation with leading space
        r"""\s+(?!\S)|\s+|"""  # Whitespace runs
        r"""\p{Extended_Pictographic}(?:\p{EMod}|\uFE0F|\u200D\p{Extended_Pictographic})*|"""  # Emoji + ZWJ + skin tones
        r""". """  # Fallback
    )

    # Production-grade Multilingual Regex Split Pattern
    # 1. Latin Contractions
    # 2. CJK Han Characters (1-2 characters per token max)
    # 3. Japanese Hiragana/Katakana (1-2 characters)
    # 4. Thai / Khmer / Lao / Myanmar syllable/grapheme clusters (consonant + combining marks)
    # 5. Indic / Dravidian / Arabic / Cyrillic words (with vowels & diacritics)
    # 6. Latin words (Capitalized, Lowercase, Titlecase)
    # 7. Numbers (up to 3 digits)
    # 8. Punctuation with leading space
    # 9. Whitespace runs / newlines
    # 10. Emojis with ZWJ and modifiers
    _compiled_regex = re.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|"""
        r"""[\p{Han}]{1,2}|"""  # CJK Han characters (1-2)
        r"""[\p{Hiragana}\p{Katakana}]{1,2}|"""  # Japanese Kana
        r"""[\p{Hangul}]{1,2}|"""  # Korean Hangul
        r"""[\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}]\p{M}*|"""  # SE Asian grapheme clusters
        r""" ?[\p{Devanagari}\p{Bengali}\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}\p{Gujarati}\p{Gurmukhi}\p{Oriya}\p{M}]+|"""  # Indic words
        r""" ?[\p{Arabic}\p{Hebrew}\p{M}]+|"""  # Arabic/Hebrew words
        r""" ?[\p{Cyrillic}\p{Greek}\p{Armenian}\p{Georgian}\p{Ethiopic}\p{M}]+|"""  # Cyrillic/Greek/Armenian/Georgian/Ethiopic
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'(?i:[sdmt]|ll|ve|re))?|"""  # Code/Latin lower/title
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'(?i:[sdmt]|ll|ve|re))?|"""  # Code/Latin uppercase
        r"""\p{N}{1,3}|"""  # Numbers
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*|"""  # Punctuation / operators
        r"""\s*[\r\n]+|"""  # Newlines
        r"""\s+(?!\S)|"""  # Trailing whitespace
        r"""\s+|"""  # Whitespace
        r"""\p{Extended_Pictographic}(?:\p{EMod}|\uFE0F|\u200D\p{Extended_Pictographic})*|"""  # Emoji
        r""". """,  # Fallback single characters
        re.UNICODE
    )

    @classmethod
    def pre_tokenize(cls, text: str) -> List[str]:
        """Pre-tokenizes multilingual text into linguistic chunks preserving script boundaries."""
        if not text:
            return []
        matches = [m.group(0) for m in cls._compiled_regex.finditer(text)]
        # Sanity check lossless roundtrip
        if "".join(matches) != text:
            # Fallback if any residual characters missed
            res = []
            last = 0
            for m in cls._compiled_regex.finditer(text):
                start, end = m.span()
                if start > last:
                    res.append(text[last:start])
                res.append(m.group(0))
                last = end
            if last < len(text):
                res.append(text[last:])
            return res
        return matches

    @staticmethod
    def verify_exact_lossless_roundtrip(original_bytes: bytes, reconstructed_bytes: bytes) -> bool:
        """Verifies 100% exact byte-for-byte roundtrip equality."""
        return original_bytes == reconstructed_bytes
