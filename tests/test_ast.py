from pathlib import Path
from semindex.ast_py import parse_python_symbols
from semindex.crawler import read_text


def test_parse_python_symbols_basic(tmp_path: Path):
    src = (
        '''
    class A(Base):
        """Doc for A"""
        def m(self, x):
            return x

    def f(y):
        """Doc f"""
        return A().m(y)
    '''
    ).lstrip()

    p = tmp_path / "mod.py"
    p.write_text(src, encoding="utf-8")

    text = read_text(str(p))
    symbols, calls = parse_python_symbols(str(p), text)

    kinds = {s.kind for s in symbols}
    names = {s.name for s in symbols}

    assert "module" in kinds
    assert any(s.kind == "class" and s.name.endswith("A") for s in symbols)
    assert any(s.kind in {"function", "method"} and s.name.endswith("f") for s in symbols)

    # call edge approx should include f -> A.m (approx name contains 'm')
    assert any("f" in c[0] and "m" in c[1] for c in calls)
