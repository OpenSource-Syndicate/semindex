from pathlib import Path
from semindex.crawler import iter_python_files, read_text


def test_iter_python_files(tmp_path: Path):
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("nope\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "c.py").write_text("print('c')\n", encoding="utf-8")

    files = list(iter_python_files(str(tmp_path)))
    assert len(files) == 1
    assert files[0].endswith("a.py")


def test_read_text(tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("x = 1\n", encoding="utf-8")
    assert read_text(str(p)) == "x = 1\n"
