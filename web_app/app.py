"""
Flask Backend Web Server for AGLM Multilingual Tokenizer Visualizer & Inspector.
Provides real-time token breakdown, multi-tokenizer comparison, and detailed token metadata.
"""

from typing import Dict, List, Any
import os
import sys
import json
import time
import tiktoken
from flask import Flask, request, jsonify, render_template, send_from_directory

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
from aglm_tokenizer.core.script_handlers import ScriptDetector
from aglm_tokenizer.eval.public_tokenizers import PublicTokenizerFactory

app = Flask(__name__, static_folder="static", template_folder="templates")

# Color palette for alternating token chips (accessible, harmonious pastel tones)
TOKEN_COLORS = [
    {"bg": "#e0f2fe", "text": "#0369a1", "border": "#bae6fd"},  # Sky
    {"bg": "#fef3c7", "text": "#b45309", "border": "#fde68a"},  # Amber
    {"bg": "#dcfce7", "text": "#15803d", "border": "#bbf7d0"},  # Emerald
    {"bg": "#f3e8ff", "text": "#7e22ce", "border": "#e9d5ff"},  # Purple
    {"bg": "#fee2e2", "text": "#b91c1c", "border": "#fecaca"},  # Rose
    {"bg": "#ffedd5", "text": "#c2410c", "border": "#fed7aa"},  # Orange
    {"bg": "#e0e7ff", "text": "#4338ca", "border": "#c7d2fe"},  # Indigo
    {"bg": "#ccfbf1", "text": "#0f766e", "border": "#99f6e4"},  # Teal
    {"bg": "#fce7f3", "text": "#be185d", "border": "#fbcfe8"},  # Pink
    {"bg": "#f1f5f9", "text": "#334155", "border": "#e2e8f0"},  # Slate
]

# Model registry
LOADED_MODELS: Dict[str, Any] = {}


def get_or_load_model(model_key: str) -> Any:
    if model_key in LOADED_MODELS:
        return LOADED_MODELS[model_key]

    print(f"[SERVER] Loading tokenizer model: {model_key}...")
    if model_key == "aglm_1m":
        tok = AGLMUniversalTokenizer.load("./exported_tokenizers/aglm_universal_1m")
        LOADED_MODELS[model_key] = tok
        return tok
    elif model_key == "aglm_256k":
        tok = AGLMUniversalTokenizer.load("./exported_tokenizers/aglm_universal_256k")
        LOADED_MODELS[model_key] = tok
        return tok
    elif model_key == "o200k_base":
        enc = tiktoken.get_encoding("o200k_base")
        LOADED_MODELS[model_key] = enc
        return enc
    elif model_key == "cl100k_base":
        enc = tiktoken.get_encoding("cl100k_base")
        LOADED_MODELS[model_key] = enc
        return enc
    elif model_key == "gemma2":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("unsloth/gemma-2-9b")
        LOADED_MODELS[model_key] = tok
        return tok
    else:
        # Load via transformers AutoTokenizer
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_key, trust_remote_code=True)
            LOADED_MODELS[model_key] = tok
            return tok
        except Exception as e:
            print(f"[WARN] Failed to load model {model_key}: {e}")
            raise e


AVAILABLE_MODELS_INFO = [
    {
        "id": "aglm_1m",
        "name": "AGLM Universal Max (Ours)",
        "vocab_size": "1,551,017",
        "description": "Unlimited Full-Reservoir Multilingual Tokenizer (1.55M+ Empirical Candidates)",
        "family": "Universal Multilingual",
        "badge": "1.55M Max (Winner)"
    },
    {
        "id": "o200k_base",
        "name": "OpenAI o200k_base (GPT-4o)",
        "vocab_size": "200,019",
        "description": "OpenAI GPT-4o Flagship Multilingual Production Tokenizer",
        "family": "Byte-level BPE",
        "badge": "GPT-4o"
    },
    {
        "id": "gemma2",
        "name": "Google Gemma 2",
        "vocab_size": "256,000",
        "description": "Google Gemma 2 9B SentencePiece Multilingual Tokenizer",
        "family": "SentencePiece BPE",
        "badge": "Gemma 2"
    },
    {
        "id": "aglm_256k",
        "name": "AGLM Universal 256K (Ours)",
        "vocab_size": "256,000",
        "description": "Compact 256K Balanced Production Tier with Dravidian & Indic Enhancements",
        "family": "Universal Multilingual",
        "badge": "256K Tier"
    },
    {
        "id": "cl100k_base",
        "name": "OpenAI cl100k_base (GPT-4)",
        "vocab_size": "100,277",
        "description": "OpenAI GPT-4 / ChatGPT Standard Production Tokenizer",
        "family": "Byte-level BPE",
        "badge": "GPT-4"
    }
]


def tokenize_with_model(model_key: str, text: str) -> Dict[str, Any]:
    """Tokenizes text with a specific model and returns rich token metadata."""
    if not text:
        return {
            "tokens": [],
            "token_count": 0,
            "raw_bytes": 0,
            "char_count": 0,
            "bytes_per_token": 0.0,
            "tokens_per_word": 0.0
        }

    raw_bytes = text.encode("utf-8")
    raw_b_len = len(raw_bytes)
    char_len = len(text)
    word_count = len(text.split()) or 1

    tok = get_or_load_model(model_key)
    tokens_meta = []

    # 1. Native AGLM tokenizers
    if model_key in ("aglm_1m", "aglm_256k"):
        token_ids = tok.encode(text)
        for i, tid in enumerate(token_ids):
            # Decode single token
            piece_str = tok.decode([tid])
            p_bytes = piece_str.encode("utf-8")
            st = ScriptDetector.detect_text_script(piece_str)
            tokens_meta.append({
                "index": i,
                "id": tid,
                "text": piece_str,
                "bytes_hex": p_bytes.hex(),
                "byte_len": len(p_bytes),
                "script": st.value,
                "color": TOKEN_COLORS[i % len(TOKEN_COLORS)]
            })

    # 2. Tiktoken models (o200k, cl100k)
    elif model_key in ("o200k_base", "cl100k_base"):
        token_ids = tok.encode(text, allowed_special="all")
        for i, tid in enumerate(token_ids):
            try:
                b = tok.decode_single_token_bytes(tid)
                piece_str = b.decode("utf-8", errors="replace")
                b_hex = b.hex()
                b_len = len(b)
            except Exception:
                piece_str = f"<{tid}>"
                b_hex = ""
                b_len = 1
            st = ScriptDetector.detect_text_script(piece_str)
            tokens_meta.append({
                "index": i,
                "id": tid,
                "text": piece_str,
                "bytes_hex": b_hex,
                "byte_len": b_len,
                "script": st.value,
                "color": TOKEN_COLORS[i % len(TOKEN_COLORS)]
            })

    # 3. HuggingFace wrappers (e.g. Gemma 2)
    else:
        try:
            token_ids = tok.encode(text, add_special_tokens=False)
        except Exception:
            token_ids = tok.encode(text)
        for i, tid in enumerate(token_ids):
            piece_str = tok.decode([tid])
            p_bytes = piece_str.encode("utf-8", errors="replace")
            st = ScriptDetector.detect_text_script(piece_str)
            tokens_meta.append({
                "index": i,
                "id": tid,
                "text": piece_str,
                "bytes_hex": p_bytes.hex(),
                "byte_len": len(p_bytes),
                "script": st.value,
                "color": TOKEN_COLORS[i % len(TOKEN_COLORS)]
            })

    num_tokens = len(tokens_meta)
    bpt = raw_b_len / num_tokens if num_tokens > 0 else 0.0
    tpw = num_tokens / word_count

    return {
        "tokens": tokens_meta,
        "token_count": num_tokens,
        "raw_bytes": raw_b_len,
        "char_count": char_len,
        "word_count": word_count,
        "bytes_per_token": round(bpt, 2),
        "tokens_per_word": round(tpw, 2)
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({"models": AVAILABLE_MODELS_INFO})


@app.route("/api/reload", methods=["POST", "GET"])
def api_reload():
    """Hot-reloads all models from disk into memory without killing the web process."""
    LOADED_MODELS.clear()
    try:
        get_or_load_model("aglm_1m")
        get_or_load_model("o200k_base")
        return jsonify({"status": "success", "message": "Models reloaded successfully from latest disk artifacts."})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    data = request.get_json() or {}
    text = data.get("text", "")
    primary_model = data.get("model", "aglm_1m")
    compare_models = data.get("compare_models", [])

    # Always compute primary model
    results = {
        primary_model: tokenize_with_model(primary_model, text)
    }

    # If compare models requested, compute them as well
    for m in compare_models:
        if m != primary_model and m not in results:
            try:
                results[m] = tokenize_with_model(m, text)
            except Exception as e:
                results[m] = {"error": str(e)}

    return jsonify({
        "input_text": text,
        "results": results
    })


if __name__ == "__main__":
    # Preload the 1M model on startup
    try:
        get_or_load_model("aglm_1m")
        get_or_load_model("o200k_base")
    except Exception as e:
        print(f"[WARN] Preloading models warning: {e}")

    port = 7860
    print(f"\n" + "=" * 80)
    print(f"🚀 AGLM TOKENIZER WEB VISUALIZER RUNNING ON: http://localhost:{port}")
    print(f"=" * 80 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
