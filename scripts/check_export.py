#!/usr/bin/env python
"""Re-score released sources from the tables alone and print the paper's headline number for each, so an export (or a
download) can be checked against the paper before anything else runs.

  python scripts/check_export.py docci_s2            # DOCCI: top-1 accuracy (paper Table 2: 76.2 for two sentences)
  python scripts/check_export.py --all               # every source present in the data folder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qr.data import data_root, load            # noqa: E402
from qr.score import lead, rank                 # noqa: E402

TAU = 0.114


def winoground(mode: str = "single") -> str:
    """Winoground text, image and group scores from the one yes/no question per caption-image pair."""
    import pyarrow.parquet as pq
    rows = [r for r in pq.read_table(data_root() / "winoground/answers.parquet").to_pylist() if r["mode"] == mode]
    by = {}
    for r in rows:
        by.setdefault(r["example"], {})[(r["caption"], r["image"])] = r["p_yes"]
    t = i = g = 0
    for p in by.values():
        tt = p[(0, 0)] > p[(1, 0)] and p[(1, 1)] > p[(0, 1)]      # each image prefers its own caption
        ii = p[(0, 0)] > p[(0, 1)] and p[(1, 1)] > p[(1, 0)]      # each caption prefers its own image
        t += tt; i += ii; g += tt and ii
    n = len(by)
    return f"winoground ({mode}): {n} examples, text {100 * t / n:.2f}, image {100 * i / n:.2f}, group {100 * g / n:.2f}"


def ap_at(ids: list[str], rel: set[str], k: int) -> float:
    hits = s = 0.0
    for i, x in enumerate(ids[:k], 1):
        if x in rel:
            hits += 1
            s += hits / i
    return s / max(1, min(len(rel), k))


def check(source: str, k: int | None = None, clef: bool = False) -> str:
    Q, P = load(source, answers_dir="clef_flash" if clef else None)
    top1 = sure = sure_right = 0
    aps = []
    for qid, photos in P.items():
        if k:
            photos = [p for p in photos if p.search_rank <= k]
        R = rank(photos)
        rel = set(Q[qid]["relevant"])
        ids = [p.image_id for p, _ in R]
        if Q[qid]["benchmark"] == "docci":
            ok = ids[0] in rel
            top1 += ok
            if lead(R) >= TAU:
                sure += 1
                sure_right += ok
        else:
            aps.append(ap_at(ids, rel, 100 if Q[qid]["benchmark"] == "inquire_rerank" else 50))
    n = len(P)
    if aps:
        line = f"{source}{f' @{k}' if k else ''}{' (Clef-Flash)' if clef else ''}: {n} queries, mean AP {100 * sum(aps) / n:.1f}"
        if source.startswith("inquire_rerank"):   # Divide and Conquer's subset: the queries outside INQUIRE's Species category
            sub = [x for qid, x in zip(P, aps) if json.loads(Q[qid]["meta"] or "{}").get("supercategory") != "Species"]
            line += f" | {len(sub)} queries outside Species: {100 * sum(sub) / max(1, len(sub)):.1f}"
        return line
    return (f"{source}{' (Clef-Flash)' if clef else ''}: {n} queries, top-1 {100 * top1 / n:.1f}% | lead >= {TAU}: sure for {100 * sure / n:.0f}%, "
            f"right on {100 * sure_right / max(1, sure):.1f}% of those")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-k", type=int, help="check only the search's top k (INQUIRE-Fullrank depths 100, 200, 500)")
    ap.add_argument("--clef", action="store_true", help="score the open Clef-Flash's answers instead of Jev's")
    a = ap.parse_args()
    names = sorted(p.name for p in data_root().iterdir() if (p / "queries.parquet").exists()) if a.all else a.sources
    for s in names:
        print(winoground() if s == "winoground" else check(s, a.k, a.clef))
