#!/usr/bin/env python
"""GPT-4o's answers for the paired comparison of Sec. 4, read from INQUIRE's own cache (not redistributed here).

INQUIRE's authors cached GPT-4o's response to every (photo, prompt) pair it reranked. The file is
`openai-gpt4o20240806--nosystem-cache.npy`, a pickled numpy dict, so open it only from that source. They share it on
request (their cache README, github.com/inquire-benchmark/INQUIRE, cache/, lists the full set of model caches as still to
come). This script keeps, for each photo in
`inquire_gpt4o_pool` (CLIP ViT-H's top 100 for the query as written), INQUIRE's P(yes): P(Yes) / (P(Yes) + P(No)) from
the first token's top log-probabilities, 0 for a failed request or no Yes, 1 for no No. It writes them to
`external/gpt4o_answers.parquet` in this repository, outside the data folder so that they are never uploaded with the
dataset (git ignores the file). Then `python paper/paper_numbers.py sec4` checks the paired gain.

    python scripts/gpt4o_answers.py --cache path/to/openai-gpt4o20240806--nosystem-cache.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qr.data import data_root  # noqa: E402
from qr.gpt4o import ANSWERS  # noqa: E402


def yes_prob(resp) -> float:
    try:
        tl = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception:  # noqa: BLE001 - a stored API error; INQUIRE scores it 0
        return 0.0
    y = [lp["logprob"] for lp in tl if lp["token"] == "Yes"]
    n = [lp["logprob"] for lp in tl if lp["token"] == "No"]
    if not y:
        return 0.0
    if not n:
        return 1.0
    py, pn = float(np.exp(y).sum()), float(np.exp(n).sum())
    return py / (py + pn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None, help="the data folder (default: as qr.data.data_root)")
    a = ap.parse_args()
    root = a.data or data_root()
    cache = np.load(a.cache, allow_pickle=True).item()
    by_text: dict[str, dict] = {}
    for key, resp in cache.items():
        img, prompt = key.split("--", 1)
        if 'show exactly "' not in prompt:
            continue
        text = prompt.split('show exactly "', 1)[1].rsplit('"?', 1)[0].strip().lower()
        by_text.setdefault(text, {})[img.replace("\\", "/").lstrip("./")] = resp
    texts = {}
    for split in ("test", "val"):
        t = pq.read_table(root / f"inquire_full_vith_lc_{split}/queries.parquet", columns=["query_id", "query"]).to_pydict()
        texts.update({q: s.strip().lower() for q, s in zip(t["query_id"], t["query"])})
    pool = pq.read_table(root / "inquire_gpt4o_pool/pool.parquet", columns=["query_id", "image_id", "image_path"]).to_pydict()
    rows = []
    for q, i, p in zip(pool["query_id"], pool["image_id"], pool["image_path"]):
        g = by_text.get(texts.get(q, ""))
        if g is not None and p in g:
            rows.append(dict(query_id=q, image_id=i, p_yes=yes_prob(g[p])))
    out = ANSWERS
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([("query_id", pa.string()), ("image_id", pa.string()),
                                                                ("p_yes", pa.float64())])), out, compression="zstd")
    n_q = len({r["query_id"] for r in rows})
    print(f"{len(rows):,} answers for {n_q} queries -> {out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
