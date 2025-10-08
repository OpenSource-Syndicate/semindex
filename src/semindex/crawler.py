import os
from typing import Iterable, List, Sequence

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


def iter_files(root: str, extensions: Sequence[str]) -> Iterable[str]:
    normalized_exts = {ext.lower() for ext in extensions}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in normalized_exts:
                yield os.path.join(dirpath, filename)


def iter_python_files(root: str) -> Iterable[str]:
    yield from iter_files(root, PY_EXTS)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as rf:
        return rf.read()
