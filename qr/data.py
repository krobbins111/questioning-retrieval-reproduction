"""Load the released tables (one folder per source: queries, questions, candidates, answers) into per-query photo lists
ready for qr.score.rank. The data folder is $QR_DATA, else local.json {"data": ...}, else ../questioning-retrieval-data.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

from .score import Photo

ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    if os.environ.get("QR_DATA"):
        return Path(os.environ["QR_DATA"])
    local = ROOT / "local.json"
    if local.exists():
        return Path(json.loads(local.read_text())["data"])
    return ROOT.parent / "questioning-retrieval-data"


# which Clef-Flash answers (clef_flash/answers.parquet, column `source`) belong to which released source
CLEF_SOURCE = {"docci_s1": "docci_s1", "docci_s2": "docci_s2", "docci_s3": "docci_s3", "docci_s4": "docci_s4",
               "docci_all": "docci_all", "inquire_rerank_test": "inquire_rerank",
               "inquire_full_vith_lc_test": "inquire_full", "inquire_full_siglip2_lc_test": "inquire_full"}


def meta(q: dict) -> dict:
    """A query's metadata (INQUIRE: supercategory, category, iconic group), as a dict."""
    m = q.get("meta")
    return json.loads(m) if isinstance(m, str) and m else (m or {})


def relevant(q: dict) -> set[str]:
    r = q["relevant"]
    if isinstance(r, str):
        r = json.loads(r.replace("'", '"'))
    return {str(x) for x in r}


def _rows(path: Path, columns=None, filters=None):
    """The rows of a table as tuples, column by column (far lighter than one dict per row)."""
    t = pq.read_table(path, columns=columns, filters=filters).to_pydict()
    return list(zip(*(t[c] for c in (columns or t))))


@lru_cache(maxsize=12)
def load(source: str, root: Path | None = None, verifier: str | None = None, answers_dir: str | None = None):
    """Return (queries, photos): queries = {query_id: row}, photos = {query_id: [Photo, ...] in search order}.
    verifier picks one model where a table holds several; answers_dir='clef_flash' reads the open model's answers for
    the same queries and photos (photos it was not asked about count as failed requests). Cached: do not mutate."""
    root = Path(root) if root else data_root()
    d = root / source
    Q = {r["query_id"]: r for r in pq.read_table(d / "queries.parquet").to_pylist()}
    qs = defaultdict(list)
    for r in pq.read_table(d / "questions.parquet").to_pylist():
        qs[r["query_id"]].append(r)
    filters = []
    if answers_dir == "clef_flash" and source in CLEF_SOURCE:
        filters.append(("source", "=", CLEF_SOURCE[source]))
    if verifier:
        filters.append(("verifier", "=", verifier))
    ans = defaultdict(dict)
    for qid, iid, qn, p, e, err in _rows((root / answers_dir if answers_dir else d) / "answers.parquet",
                                         ["query_id", "image_id", "question_id", "p_yes", "expected", "error"], filters or None):
        if qid in Q:
            ans[(qid, iid)][qn] = (p, e, err)
    photos = defaultdict(list)
    for qid, iid, rk, sc in _rows(d / "candidates.parquet", ["query_id", "image_id", "search_rank", "search_score"]):
        a = ans.get((qid, iid))
        roles = qs[qid]
        subs = [x for x in roles if x["role"] == "sub"]
        w = [float(x["weight"]) for x in subs]
        if not a:
            photos[qid].append(Photo(iid, rk, sc, None, [None] * len(subs), w, None, error=True))
            continue
        h = next((a[x["question_id"]][0] for x in roles if x["role"] in ("holistic", "rewrite") and x["question_id"] in a), None)
        s = [a[x["question_id"]][0] if x["question_id"] in a else None for x in subs]
        rel = next((a[x["question_id"]][1] for x in roles if x["role"] == "relevance" and x["question_id"] in a), None)
        photos[qid].append(Photo(iid, rk, sc, h, s, w, rel, error=any(v[2] for v in a.values())))
    for qid in photos:
        photos[qid].sort(key=lambda p: p.search_rank)
    return Q, dict(photos)
