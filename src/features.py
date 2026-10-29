"""Shared text normalization, tokenizer, and TF-IDF builders.

The website (docs/) reimplements normalize_text and TOKEN_PATTERN in JavaScript.
Change both together and rerun the parity test.
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer

# Words of 2+ word characters, the same as sklearn's default token_pattern.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text):
    """Lowercase, collapse runs of whitespace, and strip the ends."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def word_tfidf(ngram_range=(1, 2), min_df=1, **kwargs):
    """Word n-gram TF-IDF. Input is expected to be normalized already."""
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        **kwargs,
    )


def char_tfidf(ngram_range=(3, 5), min_df=2, **kwargs):
    """Character n-gram TF-IDF within word boundaries, which helps with typos."""
    return TfidfVectorizer(
        lowercase=False,
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        **kwargs,
    )
