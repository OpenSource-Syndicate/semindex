import os
from typing import Iterable, List

PY_EXTS = {".py"}

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "env",
    "venv",
    "node_modules",
    ".mypy_cache",
}


def iter_python_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext.lower() in PY_EXTS:
                yield os.path.join(dirpath, f)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as rf:
        return rf.read()
