"""
Universal Token Types Taxonomy and Classification.
Implements Section 6 of Mandatory Specifications:
Categorizes every token candidate into universal structural classes:
- RAW_BYTE: UTF-8 byte fallback (0x00 to 0xFF)
- UNICODE_CHAR: Single Unicode codepoints / graphemes
- SUBWORD: Subword morphological pieces (Latin, Indic, Cyrillic, etc.)
- WHOLE_WORD: Complete word boundaries
- MORPHOLOGY_UNIT: Suffixes, prefixes, roots, agglutinative affixes
- WHITESPACE_FACTOR: Whitespace attachments (' word', '\n\n', etc.)
- CASE_FACTOR: Capitalization variants (Titlecase, ALL_CAPS, camelCase)
- PUNCTUATION: Punctuation marks and combinations
- NUMBER: Numerical sequences, floats, ints
- DATE_TIME: Dates, timestamps, ISO strings
- CURRENCY: Money symbols and values ($100, €50, ₹1000)
- URL_EMAIL: Web URLs, domains, paths, email addresses
- CODE_SYNTAX: Programming tokens (def, class, const, ->, ::)
- EMOJI: Single and compound emoji sequences (ZWJ)
- MULTI_WORD_PHRASE: Common high-utility n-grams
- RECURSIVE_GRAMMAR: Structural delimiters and tags
- TRANSLITERATION_UNIT: Romanized Indic/Arabic/Cyrillic sub-syllables
"""

from enum import Enum
from dataclasses import dataclass
import regex as re
from typing import Dict, Any, List, Optional, Set
import unicodedata


class TokenType(str, Enum):
    RAW_BYTE = "RAW_BYTE"
    UNICODE_CHAR = "UNICODE_CHAR"
    SUBWORD = "SUBWORD"
    WHOLE_WORD = "WHOLE_WORD"
    MORPHOLOGY_UNIT = "MORPHOLOGY_UNIT"
    WHITESPACE_FACTOR = "WHITESPACE_FACTOR"
    CASE_FACTOR = "CASE_FACTOR"
    PUNCTUATION = "PUNCTUATION"
    NUMBER = "NUMBER"
    DATE_TIME = "DATE_TIME"
    CURRENCY = "CURRENCY"
    URL_EMAIL = "URL_EMAIL"
    CODE_SYNTAX = "CODE_SYNTAX"
    EMOJI = "EMOJI"
    MULTI_WORD_PHRASE = "MULTI_WORD_PHRASE"
    RECURSIVE_GRAMMAR = "RECURSIVE_GRAMMAR"
    TRANSLITERATION_UNIT = "TRANSLITERATION_UNIT"
    SPECIAL_CONTROL = "SPECIAL_CONTROL"


class TokenClassifier:
    """Classifies token text into universal token taxonomy with deep linguistic heuristics."""

    _RE_URL = re.compile(r'^(https?://|www\.)[^\s/$.?#].[^\s]*$', re.IGNORECASE)
    _RE_EMAIL = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    _RE_NUMBER = re.compile(r'^\d+(\.\d+)?$')
    _RE_CURRENCY = re.compile(r'^[\$\€\£\¥\₹\₽\₩\₪\₫\₴\₺\฿]\s*\d+([.,]\d+)?$|^\d+([.,]\d+)?\s*[\$\€\£\¥\₹\₽\₩\₪\₫\₴\₺\฿]$')
    _RE_DATE = re.compile(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$')
    _RE_CODE = re.compile(r'^(def|class|function|const|let|var|import|from|return|if|else|elif|while|for|break|continue|try|except|public|private|static|void|int|float|bool|string|dict|list|tuple|async|await|lambda|=>|->|::|==|!=|<=|>=|\+\+|--)$')
    _RE_PUNCT = re.compile(r'^[^\w\s]+$')
    _RE_EMOJI = re.compile(r'^\p{Emoji}+$')
    _RE_WHITESPACE = re.compile(r'^\s+$')

    @classmethod
    def classify_token(cls, token_bytes: bytes, token_str: Optional[str] = None) -> TokenType:
        """Classify a token based on its byte sequence and string representation."""
        if len(token_bytes) == 1:
            return TokenType.RAW_BYTE

        if token_str is None:
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return TokenType.RAW_BYTE

        # Check special tokens
        if token_str.startswith("<|") and token_str.endswith("|>"):
            return TokenType.SPECIAL_CONTROL

        # Pure whitespace
        if cls._RE_WHITESPACE.fullmatch(token_str):
            return TokenType.WHITESPACE_FACTOR

        # Emoji check
        if cls._RE_EMOJI.fullmatch(token_str) and any(unicodedata.category(c) == 'So' for c in token_str):
            return TokenType.EMOJI

        # URL / Email
        if cls._RE_URL.match(token_str) or cls._RE_EMAIL.match(token_str):
            return TokenType.URL_EMAIL

        # Currency
        if cls._RE_CURRENCY.match(token_str):
            return TokenType.CURRENCY

        # Date/Time
        if cls._RE_DATE.match(token_str):
            return TokenType.DATE_TIME

        # Pure Numbers
        if cls._RE_NUMBER.match(token_str):
            return TokenType.NUMBER

        # Programming Code syntax
        if cls._RE_CODE.match(token_str.strip()):
            return TokenType.CODE_SYNTAX

        # Pure Punctuation
        if cls._RE_PUNCT.match(token_str):
            return TokenType.PUNCTUATION

        # Multi-word phrase (contains internal whitespace separating word characters)
        if re.search(r'\w+\s+\w+', token_str):
            return TokenType.MULTI_WORD_PHRASE

        # Leading whitespace attached to a word (' world')
        if token_str.startswith(" ") or token_str.startswith("\n") or token_str.startswith("\t"):
            inner = token_str.lstrip()
            if len(inner) > 0 and not cls._RE_PUNCT.match(inner):
                return TokenType.WHITESPACE_FACTOR

        # Single unicode character
        if len(token_str) == 1:
            return TokenType.UNICODE_CHAR

        # Subword vs Whole Word vs Transliteration unit
        # Suffix / Prefix indicators
        if token_str.startswith("##") or token_str.startswith("@@") or token_str.endswith("@@"):
            return TokenType.MORPHOLOGY_UNIT

        # Case factor (e.g. ALL_CAPS with length > 2)
        if token_str.isupper() and len(token_str) > 1 and token_str.isalpha():
            return TokenType.CASE_FACTOR

        # Transliteration unit heuristic: e.g. 'aalu', 'bhau', 'shukr', 'yaar', 'bhai'
        if re.match(r'^(aa|ee|oo|kh|gh|ch|jh|th|dh|ph|bh|sh|zh|ng|ny|ts|shk)', token_str, re.IGNORECASE):
            return TokenType.TRANSLITERATION_UNIT

        # Default subword vs whole word
        if token_str.isalpha():
            if len(token_str) <= 3:
                return TokenType.SUBWORD
            return TokenType.WHOLE_WORD

        return TokenType.SUBWORD


@dataclass
class TokenProfile:
    token_id: int
    token_bytes: bytes
    token_str: str
    token_type: TokenType
    script: str
    primary_languages: List[str]
    corpus_frequency: int
    held_out_frequency: int
    positions_saved: int
    bytes_saved: int
    cross_language_reuse_score: float
    collision_risk: float
    vocab_cost: int = 1
