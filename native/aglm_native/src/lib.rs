use fancy_regex::Regex as FancyRegex;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use regex::Regex;

#[cfg(not(target_endian = "little"))]
compile_error!("AGLM production uint32 shards require a little-endian build target");

const NO_TOKEN: u32 = u32::MAX;

// The Python reference's `regex` wheel uses Unicode 17.0 while regex-syntax
// 0.8.11 embeds Unicode 16.0. This is the exhaustive symmetric difference for
// every boundary-relevant general-category, script, whitespace, and case-fold
// property used by the AGLM pre-tokenizer. (Extended_Pictographic is excluded:
// the earlier punctuation alternative always shadows that branch.) Documents
// containing one of these scalars are routed to the Python reference.
fn unicode_table_risk(cp: u32) -> bool {
    matches!(
        cp,
        0x295
            | 0x88F
            | 0xC5C
            | 0xCDC
            | 0x1ACF..=0x1ADD
            | 0x1AE0..=0x1AEB
            | 0xA7CE..=0xA7CF
            | 0xA7D2
            | 0xA7D4
            | 0xA7F1
            | 0xFBC3..=0xFBD2
            | 0xFD90..=0xFD91
            | 0xFDC8..=0xFDCE
            | 0x10940..=0x10959
            | 0x10EC5..=0x10EC7
            | 0x10ED0..=0x10ED8
            | 0x10EFA..=0x10EFB
            | 0x11B60..=0x11B67
            | 0x11DB0..=0x11DDB
            | 0x11DE0..=0x11DE9
            | 0x16EA0..=0x16EB8
            | 0x16EBB..=0x16ED3
            | 0x16FF2..=0x16FF6
            | 0x187F8..=0x187FF
            | 0x18D09..=0x18D1E
            | 0x18D80..=0x18DF2
            | 0x1E6C0..=0x1E6DE
            | 0x1E6E0..=0x1E6F5
            | 0x1E6FE..=0x1E6FF
            | 0x2B73A..=0x2B73F
            | 0x2CEA2..=0x2CEAD
            | 0x323B0..=0x33479
    )
}

struct TempNode {
    token: u32,
    children: Vec<(u8, u32)>,
}

#[derive(Clone, Copy)]
struct Node {
    token: u32,
    edge_start: u32,
    edge_len: u32,
}

/// Compact byte trie implementing the reference encoder's minimum-token DAG path.
#[pyclass]
struct NativeBpe {
    nodes: Vec<Node>,
    edges: Vec<(u8, u32)>,
    root_children: [u32; 256],
    pattern: FancyRegex,
    fast_pattern: Regex,
}

struct Scratch {
    costs: Vec<u32>,
    parent_pos: Vec<u32>,
    parent_token: Vec<u32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            costs: Vec::new(),
            parent_pos: Vec::new(),
            parent_token: Vec::new(),
        }
    }

    fn prepare(&mut self, length: usize) {
        self.costs.resize(length + 1, u32::MAX);
        self.parent_pos.resize(length + 1, 0);
        self.parent_token.resize(length + 1, NO_TOKEN);
        self.costs[..=length].fill(u32::MAX);
        self.parent_token[..=length].fill(NO_TOKEN);
        self.costs[0] = 0;
    }
}

impl NativeBpe {
    #[inline]
    fn child(&self, node_id: u32, byte: u8) -> Option<u32> {
        if node_id == 0 {
            let child = self.root_children[byte as usize];
            return (child != u32::MAX).then_some(child);
        }
        let node = self.nodes[node_id as usize];
        let edges =
            &self.edges[node.edge_start as usize..(node.edge_start + node.edge_len) as usize];
        // Vocabulary-derived tries have very low branching below the root; a linear
        // scan is faster than binary search for the common case.
        for &(edge_byte, child) in edges {
            if edge_byte == byte {
                return Some(child);
            }
            if edge_byte > byte {
                break;
            }
        }
        None
    }

    fn encode_one(&self, data: &[u8], output: &mut Vec<u32>, scratch: &mut Scratch) {
        if data.is_empty() {
            return;
        }
        let n = data.len();
        if n == 1 {
            output.push(data[0] as u32);
            return;
        }
        // A 1.55M-entry vocabulary contains a large fraction of complete
        // pre-tokenized chunks. A terminal reached after one trie walk is
        // provably the unique minimum-token solution, so avoid allocating and
        // solving the DAG in that common case without changing tie-breaking.
        let mut whole_node = 0u32;
        let mut whole_match = true;
        for &byte in data {
            if let Some(child) = self.child(whole_node, byte) {
                whole_node = child;
            } else {
                whole_match = false;
                break;
            }
        }
        if whole_match {
            let token = self.nodes[whole_node as usize].token;
            if token != NO_TOKEN {
                output.push(token);
                return;
            }
        }
        let unreachable = u32::MAX;
        scratch.prepare(n);
        let costs = &mut scratch.costs;
        let parent_pos = &mut scratch.parent_pos;
        let parent_token = &mut scratch.parent_token;

        for start in 0..n {
            if costs[start] == unreachable {
                continue;
            }
            let mut node_id = 0u32;
            let mut end = start;
            while end < n {
                let Some(child) = self.child(node_id, data[end]) else {
                    break;
                };
                node_id = child;
                end += 1;
                let token = self.nodes[node_id as usize].token;
                if token != NO_TOKEN {
                    let candidate = costs[start] + 1;
                    // Strictly less is required for identical reference tie-breaking.
                    if candidate < costs[end] {
                        costs[end] = candidate;
                        parent_pos[end] = start as u32;
                        parent_token[end] = token;
                    }
                }
            }
        }

        let before = output.len();
        let mut cursor = n;
        while cursor > 0 {
            let token = parent_token[cursor];
            // Base byte tokens guarantee reachability for every byte sequence.
            if token == NO_TOKEN {
                output.truncate(before);
                output.extend(data.iter().map(|value| *value as u32));
                return;
            }
            output.push(token);
            cursor = parent_pos[cursor] as usize;
        }
        output[before..].reverse();
    }

    /// Visit boundaries produced by the linear-time regex engine while restoring
    /// the sole lookahead-dependent GPT-style whitespace rule. A whitespace run
    /// before a prefixable token must leave its final ASCII space for that token.
    /// Re-running the anchored regex from that final byte is both exact and rare.
    fn fast_segments<F>(&self, text: &str, mut visit: F)
    where
        F: FnMut(&str),
    {
        let mut cursor = 0usize;
        while cursor < text.len() {
            let Some(matched) = self.fast_pattern.find_at(text, cursor) else {
                visit(&text[cursor..]);
                return;
            };
            if matched.start() > cursor {
                visit(&text[cursor..matched.start()]);
            }
            let value = matched.as_str();
            let is_whitespace_run = value.chars().all(char::is_whitespace);
            let char_count = value.chars().count();
            if is_whitespace_run
                && char_count >= 2
                && !value.contains(['\r', '\n'])
                && matched.end() < text.len()
            {
                // `\s+(?!\S)` greedily consumes all but the final whitespace
                // scalar before non-whitespace. Re-run from that scalar so an
                // earlier alternative may attach it to the following token.
                let final_char_offset = value.char_indices().next_back().unwrap().0;
                let pivot = matched.start() + final_char_offset;
                visit(&text[matched.start()..pivot]);
                cursor = pivot;
            } else {
                visit(value);
                cursor = matched.end();
            }
        }
    }

    fn encode_fast(&self, text: &str) -> Vec<u32> {
        let mut output = Vec::with_capacity(text.len() / 3 + 8);
        let mut scratch = Scratch::new();
        self.fast_segments(text, |segment| {
            self.encode_one(segment.as_bytes(), &mut output, &mut scratch)
        });
        output
    }
}

#[pymethods]
impl NativeBpe {
    #[new]
    fn new(id_to_bytes: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut vocabulary: Vec<(Vec<u8>, u32)> = Vec::with_capacity(id_to_bytes.len());
        for (token_id, token_bytes) in id_to_bytes.iter() {
            let id: u32 = token_id.extract()?;
            let bytes = token_bytes.cast::<PyBytes>()?.as_bytes().to_vec();
            vocabulary.push((bytes, id));
        }
        vocabulary.sort_unstable_by(|left, right| left.0.cmp(&right.0));

        let mut temporary = Vec::new();
        temporary.push(TempNode {
            token: NO_TOKEN,
            children: Vec::new(),
        });
        let mut previous: Vec<u8> = Vec::new();
        let mut path: Vec<u32> = vec![0];
        for (bytes, token_id) in vocabulary {
            let common = previous
                .iter()
                .zip(bytes.iter())
                .take_while(|(a, b)| a == b)
                .count();
            path.truncate(common + 1);
            for &byte in &bytes[common..] {
                let child = temporary.len() as u32;
                temporary.push(TempNode {
                    token: NO_TOKEN,
                    children: Vec::new(),
                });
                temporary[*path.last().unwrap() as usize]
                    .children
                    .push((byte, child));
                path.push(child);
            }
            temporary[*path.last().unwrap() as usize].token = token_id;
            previous = bytes;
        }

        let edge_count: usize = temporary.iter().map(|node| node.children.len()).sum();
        let mut nodes = Vec::with_capacity(temporary.len());
        let mut edges = Vec::with_capacity(edge_count);
        for mut node in temporary {
            node.children.sort_unstable_by_key(|edge| edge.0);
            let start = edges.len() as u32;
            let length = node.children.len() as u32;
            edges.extend(node.children);
            nodes.push(Node {
                token: node.token,
                edge_start: start,
                edge_len: length,
            });
        }
        let mut root_children = [u32::MAX; 256];
        let root = nodes[0];
        for &(byte, child) in
            &edges[root.edge_start as usize..(root.edge_start + root.edge_len) as usize]
        {
            root_children[byte as usize] = child;
        }
        let pattern = FancyRegex::new(concat!(
            r"'(?i:[sdmt]|ll|ve|re)|",
            r"[\p{Han}]{1,2}|",
            r"[\p{Hiragana}\p{Katakana}]{1,2}|",
            r"[\p{Hangul}]{1,2}|",
            r"[\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}]\p{M}*|",
            r" ?[\p{Devanagari}\p{Bengali}\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}\p{Gujarati}\p{Gurmukhi}\p{Oriya}\p{M}]+|",
            r" ?[\p{Arabic}\p{Hebrew}\p{M}]+|",
            r" ?[\p{Cyrillic}\p{Greek}\p{Armenian}\p{Georgian}\p{Ethiopic}\p{M}]+|",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'(?i:[sdmt]|ll|ve|re))?|",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'(?i:[sdmt]|ll|ve|re))?|",
            r"\p{N}{1,3}|",
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
            r"\s*[\r\n]+|",
            r"\s+(?!\S)|",
            r"\s+|",
            r"\p{Extended_Pictographic}(?:\p{EMod}|\u{FE0F}|\u{200D}\p{Extended_Pictographic})*|",
            r". "
        )).map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        // Diagnostic DFA/NFA-compatible version. The negative lookahead branch is
        // deliberately collapsed here; Python tests compare its boundaries before
        // this can ever become a production path.
        let fast_pattern = Regex::new(concat!(
            r"'(?i:[sdmt]|ll|ve|re)|",
            r"[\p{Han}]{1,2}|",
            r"[\p{Hiragana}\p{Katakana}]{1,2}|",
            r"[\p{Hangul}]{1,2}|",
            r"[\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}]\p{M}*|",
            r" ?[\p{Devanagari}\p{Bengali}\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}\p{Gujarati}\p{Gurmukhi}\p{Oriya}\p{M}]+|",
            r" ?[\p{Arabic}\p{Hebrew}\p{M}]+|",
            r" ?[\p{Cyrillic}\p{Greek}\p{Armenian}\p{Georgian}\p{Ethiopic}\p{M}]+|",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'(?i:[sdmt]|ll|ve|re))?|",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'(?i:[sdmt]|ll|ve|re))?|",
            r"\p{N}{1,3}|",
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
            r"\s*[\r\n]+|",
            r"\s+|",
            r"\p{Extended_Pictographic}(?:\p{EMod}|\u{FE0F}|\u{200D}\p{Extended_Pictographic})*|",
            r". "
        )).map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            nodes,
            edges,
            root_children,
            pattern,
            fast_pattern,
        })
    }

    fn encode_segments(&self, segments: Vec<String>) -> Vec<u32> {
        let estimated: usize = segments.iter().map(|value| value.len()).sum();
        let mut output = Vec::with_capacity(estimated / 3 + 8);
        let mut scratch = Scratch::new();
        for segment in segments {
            self.encode_one(segment.as_bytes(), &mut output, &mut scratch);
        }
        output
    }

    fn requires_reference_fallback(&self, text: &str) -> bool {
        text.chars().any(|value| unicode_table_risk(value as u32))
    }

    fn unicode_table_risk_count(&self) -> usize {
        (0..=0x10FFFF)
            .filter(|value| unicode_table_risk(*value))
            .count()
    }

    fn encode_segments_u32<'py>(
        &self,
        py: Python<'py>,
        segments: Vec<String>,
    ) -> Bound<'py, PyBytes> {
        let ids = self.encode_segments(segments);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                ids.as_ptr() as *const u8,
                ids.len() * std::mem::size_of::<u32>(),
            )
        };
        PyBytes::new(py, bytes)
    }

    fn encode_text(&self, text: &str) -> PyResult<Vec<u32>> {
        let mut output = Vec::with_capacity(text.len() / 3 + 8);
        let mut scratch = Scratch::new();
        let mut last = 0usize;
        for found in self.pattern.find_iter(text) {
            let matched = found
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
            if matched.start() > last {
                self.encode_one(
                    &text.as_bytes()[last..matched.start()],
                    &mut output,
                    &mut scratch,
                );
            }
            self.encode_one(matched.as_str().as_bytes(), &mut output, &mut scratch);
            last = matched.end();
        }
        if last < text.len() {
            self.encode_one(&text.as_bytes()[last..], &mut output, &mut scratch);
        }
        Ok(output)
    }

    fn encode_text_fast(&self, py: Python<'_>, text: String) -> Vec<u32> {
        py.detach(|| self.encode_fast(&text))
    }

    fn encode_text_fast_u32<'py>(&self, py: Python<'py>, text: String) -> Bound<'py, PyBytes> {
        let ids = py.detach(|| self.encode_fast(&text));
        let bytes = unsafe {
            std::slice::from_raw_parts(
                ids.as_ptr() as *const u8,
                ids.len() * std::mem::size_of::<u32>(),
            )
        };
        PyBytes::new(py, bytes)
    }

    /// Exact minimum-token segmentation over the complete UTF-8 byte string.
    /// Unlike the production path, this deliberately imposes no regex/script
    /// boundaries. It is a read-only diagnostic over the existing vocabulary.
    fn encode_text_minimal(&self, py: Python<'_>, text: String) -> Vec<u32> {
        py.detach(|| {
            let mut output = Vec::with_capacity(text.len() / 3 + 8);
            let mut scratch = Scratch::new();
            self.encode_one(text.as_bytes(), &mut output, &mut scratch);
            output
        })
    }

    /// Little-endian uint32 form of `encode_text_minimal` for streaming audits.
    fn encode_text_minimal_u32<'py>(&self, py: Python<'py>, text: String) -> Bound<'py, PyBytes> {
        let ids = py.detach(|| {
            let mut output = Vec::with_capacity(text.len() / 3 + 8);
            let mut scratch = Scratch::new();
            self.encode_one(text.as_bytes(), &mut output, &mut scratch);
            output
        });
        let bytes = unsafe {
            std::slice::from_raw_parts(
                ids.as_ptr() as *const u8,
                ids.len() * std::mem::size_of::<u32>(),
            )
        };
        PyBytes::new(py, bytes)
    }

    fn pre_tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        let mut output = Vec::new();
        let mut last = 0usize;
        for found in self.pattern.find_iter(text) {
            let matched = found
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
            if matched.start() > last {
                output.push(text[last..matched.start()].to_owned());
            }
            output.push(matched.as_str().to_owned());
            last = matched.end();
        }
        if last < text.len() {
            output.push(text[last..].to_owned());
        }
        Ok(output)
    }

    fn pre_tokenize_fast(&self, text: &str) -> Vec<String> {
        let mut output = Vec::new();
        let mut last = 0usize;
        for matched in self.fast_pattern.find_iter(text) {
            if matched.start() > last {
                output.push(text[last..matched.start()].to_owned());
            }
            output.push(matched.as_str().to_owned());
            last = matched.end();
        }
        if last < text.len() {
            output.push(text[last..].to_owned());
        }
        output
    }

    fn pre_tokenize_fast_exact(&self, text: &str) -> Vec<String> {
        let mut output = Vec::new();
        self.fast_segments(text, |segment| output.push(segment.to_owned()));
        output
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[pymodule]
fn _aglm_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeBpe>()?;
    Ok(())
}
