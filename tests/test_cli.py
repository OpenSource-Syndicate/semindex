import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

import semindex.cli as cli


class FakeModel:
    class Cfg:
        hidden_size = 8

    config = Cfg()


class FakeEmbedder:
    def __init__(self, *a, **kw):
        self.model = FakeModel()

    def encode(self, texts, batch_size=16, max_length=512):
        # deterministic tiny embeddings
        vecs = []
        for i, _t in enumerate(texts):
            rng = np.random.default_rng(seed=1234 + i)
            v = rng.standard_normal(self.model.config.hidden_size).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            vecs.append(v)
        return np.vstack(vecs)


def run_cli(args):
    argv = ["semindex"] + args
    old = sys.argv
    try:
        sys.argv = argv
        cli.main()
    finally:
        sys.argv = old


def test_index_and_query_with_fake_embedder(tmp_path: Path, monkeypatch):
    # Create a tiny repo
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text(
        """
class Greeter:
    def hello(self, name):
        return f"Hello, {name}!"

def greet(name):
    return Greeter().hello(name)
""".lstrip(),
        encoding="utf-8",
    )

    # Use a temporary index dir
    idx = tmp_path / ".semindex"

    # Monkeypatch Embedder to avoid model download
    monkeypatch.setattr(cli, "Embedder", FakeEmbedder)

    # Index
    run_cli(["index", str(repo), "--index-dir", str(idx)])

    # Query
    run_cli(["query", "greet someone", "--index-dir", str(idx)])

    # Basic assertions: index files created
    assert (idx / "index.faiss").exists()
    assert (idx / "semindex.db").exists()

    with sqlite3.connect(idx / "semindex.db") as con:
        cur = con.cursor()

        file_languages = {row[0] for row in cur.execute("SELECT language FROM files").fetchall()}
        assert file_languages == {"python"}

        symbol_row = cur.execute(
            """
            SELECT language, namespace, symbol_type
            FROM symbols
            WHERE name LIKE ?
            ORDER BY id DESC
            LIMIT 1
            """,
            ("Greeter.hello%",),
        ).fetchone()
        assert symbol_row is not None
        language, namespace, symbol_type = symbol_row
        assert language == "python"
        assert namespace == "Greeter"
        assert symbol_type == "method"
