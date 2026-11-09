"""Checks for the website's in-browser model (docs/js/model.js) and the exported files.

There is no JavaScript runtime in the test environment, so this file checks two things:
  1. A line-by-line Python mirror of model.js's n-gram code gives exactly sklearn's features on
     every training message, and model.js's character class [\\p{L}\\p{N}_] matches the same
     characters as Python's \\w on every character in the data.
  2. The exported files are consistent (sizes, int8 agreement).
The JavaScript itself is checked against Python in the browser by docs/parity.html.
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED, DOCS_ASSETS
from src.features import TOKEN_PATTERN, char_tfidf, normalize_text, word_tfidf


def js_word_tokens(norm):
    return re.findall(r"\w{2,}", norm)                 # model.js: /[\p{L}\p{N}_]{2,}/gu


def js_word_ngrams(toks, lo, hi):
    return [" ".join(toks[i:i + n]) for n in range(lo, hi + 1) for i in range(len(toks) - n + 1)]


def js_char_wb(norm, lo, hi):
    out = []
    for word in [w for w in norm.split(" ") if w]:
        w = " " + word + " "
        for n in range(lo, hi + 1):
            offset = 0
            out.append(w[:n])
            while offset + n < len(w):
                offset += 1
                out.append(w[offset:offset + n])
            if offset == 0:
                break
    return out


@pytest.fixture(scope="module")
def texts():
    return pd.read_csv(DATA_PROCESSED / "train_split.csv")["text"].tolist()


def test_js_character_class_matches_python_word_chars(texts):
    chars = set("".join(texts).lower())
    js_class = lambda ch: unicodedata.category(ch)[0] in "LN" or ch == "_"
    assert all(bool(re.match(r"\w", ch)) == js_class(ch) for ch in chars)


def test_js_ngram_logic_matches_sklearn(texts):
    norm = [normalize_text(t) for t in texts]
    word = word_tfidf().build_analyzer()
    char = char_tfidf().build_analyzer()
    for t in norm:
        assert js_word_tokens(t) == re.findall(TOKEN_PATTERN, t)
        assert sorted(js_word_ngrams(js_word_tokens(t), 1, 2)) == sorted(word(t))
        assert js_char_wb(t, 3, 5) == char(t)


@pytest.mark.skipif(not (DOCS_ASSETS / "model.json").exists(), reason="run python -m src.export_web first")
def test_exported_files_are_consistent():
    meta = json.loads((DOCS_ASSETS / "model.json").read_text())
    n_feat = len(meta["word"]["vocab"]) + len(meta["char"]["vocab"])
    assert meta["n_features"] == n_feat == len(meta["scale"])
    assert len(meta["word"]["idf"]) == len(meta["word"]["vocab"]) and len(meta["char"]["idf"]) == len(meta["char"]["vocab"])
    assert len(meta["classes"]) == len(meta["intercept"]) == 77
    assert (DOCS_ASSETS / "weights.bin").stat().st_size == n_feat * 77
    parity = json.loads((DOCS_ASSETS / "parity.json").read_text())
    assert parity["quantization_top1_agreement"] == 1.0
    assert np.allclose(np.sum(parity["p_quant"], axis=1), 1, atol=1e-4)
