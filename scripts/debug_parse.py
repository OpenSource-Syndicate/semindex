from semindex.ast_py import parse_python_symbols

source = """\nclass Greeter:\n    def hello(self, name):\n        return f\"Hello, {name}!\"\n\ndef greet(name):\n    return Greeter().hello(name)\n""".lstrip()

symbols, calls = parse_python_symbols("m.py", source)

for sym in symbols:
    print(f"name={sym.name!r}, kind={sym.kind}, namespace={sym.namespace!r}, symbol_type={sym.symbol_type!r}, language={sym.language!r}")

print("calls:")
for caller, callee in calls:
    print(f"  {caller} -> {callee}")
