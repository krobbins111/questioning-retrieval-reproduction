"""GPT-4o's published reranking of INQUIRE-Fullrank, query by query, for the paired comparison of Sec. 4.

GPT-4o's answers belong to INQUIRE (Vendrow et al., 2024). Its authors share them on request as a numpy cache, and
this repository does not redistribute them. `scripts/gpt4o_answers.py` reads that cache into `external/gpt4o_answers.parquet`
in this repository (outside the data folder, so that it is never uploaded with the dataset; git ignores it): P(yes) for
each photo GPT-4o reranked, which are the photos in `inquire_gpt4o_pool` (CLIP ViT-H's top 100 for the query as
written). Without that file, everything here returns None.

GPT-4o's ranking follows INQUIRE's two-stage script: `np.flip(np.argsort(p_yes))` over the photos in CLIP's order, and
the top 50 are scored with AP@50. Where answers tie, the order is whatever numpy's argsort leaves. Across numpy versions
this moves GPT-4o's mean by about 0.001 points.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from qr.data import data_root, load, relevant
from qr.metrics import ap_at

ANSWERS = Path(__file__).resolve().parent.parent / "external" / "gpt4o_answers.parquet"


def gpt4o_ap(split: str = "test", root: Path | None = None) -> dict[str, float] | None:
    """query_id -> GPT-4o's AP@50 on the queries whose 100 photos all have an answer (160 test, 40 validation)."""
    root = Path(root) if root else data_root()
    f = ANSWERS
    if not f.exists():
        return None
    ans = {(q, i): p for q, i, p in zip(*pq.read_table(f, columns=["query_id", "image_id", "p_yes"]).to_pydict().values())}
    pool = defaultdict(list)
    t = pq.read_table(root / "inquire_gpt4o_pool/pool.parquet", columns=["query_id", "split", "rank", "image_id"]).to_pydict()
    for q, s, r, i in zip(t["query_id"], t["split"], t["rank"], t["image_id"]):
        if s == split:
            pool[q].append((r, i))
    Q, _ = load(f"inquire_full_vith_lc_{split}", root)
    out = {}
    for q, items in sorted(pool.items()):
        ids = [i for _, i in sorted(items)]
        if q not in Q or not all((q, i) in ans for i in ids):
            continue
        y = np.array([ans[(q, i)] for i in ids], float)
        out[q] = ap_at([ids[j] for j in np.flip(np.argsort(y))], relevant(Q[q]), 50)
    return out


class Nov14Stream:
    """Bootstrap intervals drawn as in the analysis that produced Sec. 4's paired interval (+0.9 to +4.4).

    That analysis drew every interval from one generator, numpy's default_rng(0), with 4,000 resamples and the 2.5 and
    97.5 percentiles. For each depth (100, 200, then 500 photos, test queries in id order), it made five draws:
    1-3. the ViT-H pool scored with ViT-H's own search score, against SigLIP2's score (200 queries, three times);
    4. the paired difference with GPT-4o (the 160 queries with its answers);
    5. the same with SigLIP2's score.

    Sec. 4 prints draw 4 at 100 photos. The other draws are replayed so that the printed interval comes out exactly.
    The supplement's paired table continues the same generator for the SigLIP2 pool."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def draw(self, d) -> tuple[float, float, float]:
        d = np.asarray(d, float)
        bs = d[self.rng.integers(0, len(d), (4000, len(d)))].mean(1)
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    def depth(self, own: np.ndarray, sig: np.ndarray, gpt: np.ndarray, paired: np.ndarray) -> tuple[float, float, float]:
        """own, sig: AP@50 of the 200 test queries (ViT-H pool, ViT-H's or SigLIP2's search score); gpt: GPT-4o's AP@50
        where `paired` is True. Returns the paired difference with GPT-4o (mean, low, high)."""
        for _ in range(3):
            self.draw(own - sig)
        out = self.draw(own[paired] - gpt)
        self.draw(sig[paired] - gpt)
        return out
