#!/usr/bin/env python
"""Look for anything that would identify the authors before a folder is made public: names, affiliations, account
names, personal paths, e-mail addresses and access keys. Generic patterns are built in; the identifying terms
themselves are read from anon_terms.txt (one per line, case-insensitive, never committed), so this script carries none.

    python scripts/anon_scan.py                         # the repository (text files)
    python scripts/anon_scan.py --root ../questioning-retrieval-anon
    python scripts/anon_scan.py --data ../questioning-retrieval-data      # every string in every parquet table

In the tables, a term counts only where it stands apart from neighbouring letters, so a benchmark's own text that
happens to contain one inside a longer word (an iNat24 species name, say) is not reported. Exit status 1 if anything
is found.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = {".py", ".md", ".txt", ".json", ".bat", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".tex", ".csv", ".ipynb"}
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv"}
GENERIC = [r"[A-Za-z]:\\Users\\", r"/Users/[A-Za-z]", r"/home/(?!claude\b)[a-z]", r"/session[s]/", r"One[D]rive",
           r"[\w.+-]+@[\w-]+\.(?:com|edu|org|net|io)", r"\bhf_[A-Za-z0-9]{20,}", r"\bsk-[A-Za-z0-9_-]{20,}",
           r"api[_-]?key\s*[:=]\s*['\"][^'\"]{8,}", r"[Bb]earer\s+[A-Za-z0-9._-]{16,}"]


def patterns(terms_file: Path, apart: bool = False) -> re.Pattern:
    """The generic patterns and the terms; apart: a term must not touch a letter on either side."""
    terms = []
    if terms_file.exists():
        terms = [re.escape(t.strip()) for t in terms_file.read_text(encoding="utf-8").splitlines() if t.strip() and not t.startswith("#")]
    elif not apart:
        print(f"note: {terms_file.name} not found, so only generic patterns are checked (names and accounts are not)")
    if apart:
        terms = [rf"(?<![A-Za-z]){t}(?![A-Za-z])" for t in terms]
    return re.compile("|".join(GENERIC + terms), re.IGNORECASE)


def scan_text(root: Path, pat: re.Pattern) -> int:
    n = 0
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in TEXT or SKIP_DIRS & set(f.relative_to(root).parts) or f.name == "anon_terms.txt":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in pat.finditer(line):
                print(f"{f.relative_to(root)}:{i}: {m.group(0)!r}")
                n += 1
    return n


def scan_data(root: Path, pat: re.Pattern, cells: re.Pattern) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = 0
    for f in sorted(root.rglob("*.parquet")):
        t = pq.read_table(f)
        for c in t.column_names:
            typ = t.schema.field(c).type
            if not (pa.types.is_string(typ) or (pa.types.is_list(typ) and pa.types.is_string(typ.value_type))):
                continue
            for v in t.column(c).to_pylist():
                for s in (v if isinstance(v, list) else [v]):
                    if s and cells.search(s):
                        print(f"{f.relative_to(root)} [{c}]: {cells.search(s).group(0)!r}")
                        n += 1
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in TEXT:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                for m in pat.finditer(line):
                    print(f"{f.relative_to(root)}:{i}: {m.group(0)!r}")
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--data", type=Path, help="a data folder: scan its parquet tables and text files instead")
    ap.add_argument("--terms", type=Path, default=ROOT / "anon_terms.txt")
    a = ap.parse_args()
    pat = patterns(a.terms)
    n = scan_data(a.data, pat, patterns(a.terms, apart=True)) if a.data else scan_text(a.root, pat)
    print(f"{n} identifying string{'s' if n != 1 else ''} in {a.data or a.root}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
