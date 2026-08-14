"""
High-Performance BPE / Subword Core Engine.
Implements lossless encoding, 256-byte fallback, fast trie lookup, and greedy subword tokenization.
"""

from typing import Dict, List, Tuple, Optional, Set, Union
import time
import struct
from aglm_tokenizer.core.script_handlers import ScriptSegmenter


class TrieNode:
    __slots__ = ('children', 'token_id', 'is_end')

    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        self.token_id: Optional[int] = None
        self.is_end: bool = False


class ByteTrie:
    """Fast prefix trie for byte sequences."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, byte_seq: bytes, token_id: int) -> None:
        curr = self.root
        for b in byte_seq:
            if b not in curr.children:
                curr.children[b] = TrieNode()
            curr = curr.children[b]
        curr.token_id = token_id
        curr.is_end = True

    def all_prefix_matches(self, data: bytes, start_idx: int) -> List[Tuple[int, int]]:
        """
        Finds all tokens matching prefixes of data[start_idx:].
        Returns list of (token_id, matched_length).
        """
        curr = self.root
        n = len(data)
        matches: List[Tuple[int, int]] = []
        idx = start_idx

        while idx < n and data[idx] in curr.children:
            curr = curr.children[data[idx]]
            idx += 1
            if curr.is_end:
                matches.append((curr.token_id, idx - start_idx))

        if not matches:
            matches.append((data[start_idx], 1))
        return matches

    def longest_prefix_match(self, data: bytes, start_idx: int) -> Tuple[int, int]:
        """
        Finds the longest token matching data[start_idx:].
        Returns (token_id, matched_length).
        If no multi-byte match, returns byte fallback (token_id for byte, 1).
        """
        curr = self.root
        n = len(data)
        longest_token_id: Optional[int] = None
        longest_len = 0

        idx = start_idx
        while idx < n and data[idx] in curr.children:
            curr = curr.children[data[idx]]
            idx += 1
            if curr.is_end:
                longest_token_id = curr.token_id
                longest_len = idx - start_idx

        if longest_token_id is not None:
            return (longest_token_id, longest_len)
        else:
            # Fallback to single byte
            return (data[start_idx], 1)


class BPEEngine:
    """
    Universal Multilingual BPE Engine.
    Guarantees:
    1. Base 256 byte tokens for 100% exact lossless recovery.
    2. Longest prefix matching with priority merges.
    3. High-speed encode & decode.
    """

    def __init__(self):
        # 0-255 are reserved for individual bytes
        self.id_to_bytes: Dict[int, bytes] = {b: bytes([b]) for b in range(256)}
        self.bytes_to_id: Dict[bytes, int] = {bytes([b]): b for b in range(256)}
        self.special_tokens: Dict[str, int] = {}
        self.special_id_to_str: Dict[int, str] = {}
        self.trie = ByteTrie()

        # Populate trie with base bytes
        for b in range(256):
            self.trie.insert(bytes([b]), b)

        self._next_id = 256

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_bytes) + len(self.special_tokens)

    def add_token(self, token_bytes: bytes, token_id: Optional[int] = None) -> int:
        """Adds a multi-byte token to the vocabulary."""
        if token_bytes in self.bytes_to_id:
            return self.bytes_to_id[token_bytes]

        if token_id is None:
            token_id = self._next_id
            self._next_id += 1
        else:
            if token_id >= self._next_id:
                self._next_id = token_id + 1

        self.bytes_to_id[token_bytes] = token_id
        self.id_to_bytes[token_id] = token_bytes
        self.trie.insert(token_bytes, token_id)
        return token_id

    def add_special_token(self, token_str: str, token_id: Optional[int] = None) -> int:
        """Adds a special control token."""
        if token_str in self.special_tokens:
            return self.special_tokens[token_str]

        if token_id is None:
            token_id = self._next_id
            self._next_id += 1
        else:
            if token_id >= self._next_id:
                self._next_id = token_id + 1

        self.special_tokens[token_str] = token_id
        self.special_id_to_str[token_id] = token_str
        return token_id

    def encode_segment(self, segment_bytes: bytes) -> Tuple[List[int], int]:
        """
        Encodes a single chunk/segment into token IDs using minimum-token shortest path DP.
        Returns (list_of_token_ids, byte_fallback_count).
        """
        n = len(segment_bytes)
        if n == 0:
            return [], 0

        # Fast path for very short single character/byte
        if n == 1:
            tok_id, _ = self.trie.longest_prefix_match(segment_bytes, 0)
            return [tok_id], (1 if tok_id < 256 and n > 1 else 0)

        # Dynamic Programming Shortest Path (DAG Minimum Token Count)
        dp = [float('inf')] * (n + 1)
        parent: List[Optional[Tuple[int, int, int]]] = [None] * (n + 1)
        dp[0] = 0.0

        for i in range(n):
            if dp[i] == float('inf'):
                continue
            matches = self.trie.all_prefix_matches(segment_bytes, i)
            for tok_id, length in matches:
                cost = dp[i] + 1.0
                if cost < dp[i + length]:
                    dp[i + length] = cost
                    parent[i + length] = (i, tok_id, length)

        tokens: List[int] = []
        fallbacks = 0
        curr = n
        while curr > 0 and parent[curr]:
            prev, tok_id, match_len = parent[curr]
            tokens.append(tok_id)
            if match_len == 1 and tok_id < 256 and n > 1:
                fallbacks += 1
            curr = prev

        tokens.reverse()
        return tokens, fallbacks

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> Tuple[List[int], int]:
        """
        Encodes full multilingual text.
        Returns (tokens, byte_fallback_count).
        """
        if not text:
            return [], 0

        # Split text into linguistic script chunks
        chunks = ScriptSegmenter.pre_tokenize(text)
        all_tokens: List[int] = []
        total_byte_fallbacks = 0

        for chunk in chunks:
            if allowed_special and chunk in self.special_tokens and chunk in allowed_special:
                all_tokens.append(self.special_tokens[chunk])
                continue

            chunk_bytes = chunk.encode("utf-8")
            toks, fallbacks = self.encode_segment(chunk_bytes)
            all_tokens.extend(toks)
            total_byte_fallbacks += fallbacks

        return all_tokens, total_byte_fallbacks

    def decode_to_bytes(self, tokens: List[int]) -> bytes:
        """Decodes token IDs directly into raw bytes."""
        byte_chunks: List[bytes] = []
        for tid in tokens:
            if tid in self.id_to_bytes:
                byte_chunks.append(self.id_to_bytes[tid])
            elif tid in self.special_id_to_str:
                byte_chunks.append(self.special_id_to_str[tid].encode("utf-8"))
            else:
                # Unknown fallback
                byte_chunks.append(b"")
        return b"".join(byte_chunks)

    def decode(self, tokens: List[int], errors: str = "replace") -> str:
        """Decodes token IDs into a UTF-8 string."""
        raw_bytes = self.decode_to_bytes(tokens)
        return raw_bytes.decode("utf-8", errors=errors)
