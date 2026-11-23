#!/usr/bin/env python
"""Make the review copy of this repository: every file except the internal ones (the release plan, the supplement
notes, local settings, the terms list), plus ANONYMIZATION.md; then scan the copy with scripts/anon_scan.py. The copy
is what goes to GitHub and to the anonymous mirror.

    python scripts/make_anon_copy.py                     # into ../questioning-retrieval-anon, updated in place (its .git is kept)
    python scripts/make_anon_copy.py --out <folder>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTERNAL = {"PLAN.md", "supplement", "anon_terms.txt", "local.json", "external", ".git", "__pycache__", "venv", ".venv", "_to_delete"}
NOTE = """This is the anonymised review copy of the repository. Author names, affiliations, account names, personal paths
and internal planning documents were removed for double-blind review; the code, configuration and data manifest are
unchanged. The answers are downloaded from an anonymous Hugging Face account (configs/release.json).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT.parent / "questioning-retrieval-anon")
    a = ap.parse_args()
    out = a.out.resolve()
    if out == ROOT or ROOT in out.parents:
        sys.exit("the copy must be outside the repository")
    n, keep = 0, {Path("ANONYMIZATION.md")}
    for f in sorted(ROOT.rglob("*")):
        rel = f.relative_to(ROOT)
        if not f.is_file() or INTERNAL & set(rel.parts) or f.suffix in (".pyc", ".parquet", ".log", ".part"):
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        keep.add(rel)
        n += 1
    (out / "ANONYMIZATION.md").write_text(NOTE, encoding="utf-8")
    # files left from an earlier copy that the repository no longer has (the .git folder of the copy is kept)
    for f in sorted(out.rglob("*")):
        rel = f.relative_to(out)
        if f.is_file() and rel not in keep and not {".git", "__pycache__"} & set(rel.parts):
            try:
                f.unlink()
                print(f"removed {rel} (no longer in the repository)")
            except OSError:
                print(f"remove by hand: {rel} (no longer in the repository)")
    print(f"copied {n} files to {out}")
    return subprocess.call([sys.executable, str(ROOT / "scripts/anon_scan.py"), "--root", str(out),
                            "--terms", str(ROOT / "anon_terms.txt")])


if __name__ == "__main__":
    sys.exit(main())
