"""Guard: only src/evaluate_final.py may load the test set.

Looks for code that refers to the test file (the TEST_CSV constant or a quoted
"test.csv" path), not for mentions in docstrings or comments.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
ALLOWED = {"evaluate_final.py", "config.py", "download_data.py"}


def references_test_set(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.alias)) and "TEST_CSV" in {getattr(node, "id", None), getattr(node, "name", None)}:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip().endswith("test.csv"):
            return True
    return False


def test_only_evaluate_final_reads_test_set():
    offenders = [p.name for p in SRC.glob("*.py") if p.name not in ALLOWED and references_test_set(p)]
    assert not offenders, f"These modules reference the test set: {offenders}"


def test_guard_catches_a_real_reference(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('"""Never reads test.csv."""\nfrom src.config import TEST_CSV\n')
    good = tmp_path / "good.py"
    good.write_text('"""Never reads test.csv."""\nx = 1\n')
    assert references_test_set(bad)
    assert not references_test_set(good)
