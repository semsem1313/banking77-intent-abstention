"""Guard: only src/evaluate_final.py may reference the test set."""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
ALLOWED = {"evaluate_final.py", "config.py", "download_data.py"}


def test_only_evaluate_final_reads_test_set():
    offenders = [
        p.name for p in SRC.glob("*.py")
        if p.name not in ALLOWED and ("TEST_CSV" in p.read_text() or "test.csv" in p.read_text())
    ]
    assert not offenders, f"These modules reference the test set: {offenders}"
