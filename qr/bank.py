"""The larger question set of Sec. 4 ("the bank"): reading its answers and scoring them as the analysis that produced
the paper's numbers did (scripts/bank_report.py of the run).

Families of questions, all about one photo and asked in one request:
  W  the whole query, in three wordings          R  requirements (statements), with required/decisive flags
  C  details, multiple choice ("cannot tell")    N  the likely alternative as a statement (its answer reversed)
  M  match or near-miss, multiple choice         V  1-to-5 match
  X  what is wrong, multiple choice              F  descriptors (counts, setting): never used for the match

Each question's `value` is on a 0-to-1 scale where 1 speaks for its target. A family is reduced to one number per photo
(R: weighted geometric mean of the required statements; the others: the mean), and the bank's score is the mean of
the families. Every comparison adds the search score at 0.2, as Eq. 2 does. Missing answers take the pool's mean.
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from qr.data import data_root

FAM = ("W", "R", "C", "N", "M", "V", "X")


def fill(V):
    """A photo without an answer gets the pool's mean answer to that question (0.5 if nobody answered)."""
    V = np.array(V, float)
    if V.ndim == 1:
        V = V[:, None]
    m = np.where(np.isnan(V).all(0), 0.5, np.nanmean(np.where(np.isnan(V).all(0), 0.5, V), 0))
    return np.where(np.isnan(V), m, V)


def geo(P, w=None, floor: float = 0.02):
    P = np.clip(fill(P), floor, 1)
    w = np.ones(P.shape[1]) if w is None else np.asarray(w, float)
    return np.exp((np.log(P) * w).sum(1) / max(w.sum(), 1e-9))


def mm(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.full(len(x), 0.5) if not np.isfinite(hi - lo) or hi - lo < 1e-9 else np.nan_to_num((x - lo) / (hi - lo), nan=0.0)


def boot(d, n: int = 4000, seed: int = 0):
    """The analysis's paired bootstrap: 4,000 resamples of the queries, percentiles 2.5 and 97.5."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if not len(d):
        return float("nan"), float("nan"), float("nan")
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


@lru_cache(maxsize=4)
def load(source: str, root: Path | None = None) -> list[dict]:
    """One dict per query, in query-id order: ids, labels, search score, the earlier run's seven answers, and each pass's
    questions with their values and option probabilities."""
    d = Path(root) if root else data_root()
    d = d / source
    Q = {r["query_id"]: r for r in pq.read_table(d / "queries.parquet").to_pylist()}
    ph = defaultdict(list)
    for r in pq.read_table(d / "photos.parquet").to_pylist():
        ph[r["query_id"]].append(r)
    qs = defaultdict(lambda: defaultdict(list))
    for r in pq.read_table(d / "questions.parquet").to_pylist():
        qs[r["query_id"]][r["pass"]].append(r)
    val = defaultdict(dict)
    t = pq.read_table(d / "answers.parquet").to_pydict()
    for q, p, i, k, v, pr in zip(t["query_id"], t["pass"], t["image_id"], t["question_id"], t["value"], t["probs"]):
        val[(q, p, k)][i] = (v, pr)
    rows = []
    for q in sorted(Q):
        P = sorted(ph[q], key=lambda r: r["rank"])
        ids = [r["image_id"] for r in P]
        nan = lambda x: np.nan if x is None else x  # noqa: E731
        S = np.array([[nan(x) for x in (r["earlier_s"] or [])] for r in P], float).reshape(len(P), -1)
        passes = {}
        for p, qds in qs[q].items():
            v = {x["question_id"]: np.array([nan(val[(q, p, x["question_id"])].get(i, (None, None))[0]) for i in ids], float) for x in qds}
            prob = {x["question_id"]: np.array([val[(q, p, x["question_id"])].get(i, (None, None))[1] or [np.nan] * len(x["options"] or x["levels"] or [])
                                                for i in ids], float) for x in qds if x["kind"] != "noul"}
            passes[p] = {"qds": qds, "val": v, "prob": prob}
        rows.append({"query_id": q, "query": Q[q]["query"], "ids": ids, "y": np.array([r["relevant"] for r in P]), "nrel": Q[q]["n_relevant"],
                     "metric": Q[q]["metric"], "sim": np.array([r["search_score"] for r in P], float),
                     "old": {"h": np.array([nan(r["earlier_h"]) for r in P], float), "r": np.array([nan(r["earlier_r"]) for r in P], float),
                             "S": S, "w": np.array(Q[q]["earlier_w"] or [], float)},
                     "passes": passes})
    return rows


def metric_of(rows: list[dict]):
    """R@1 when every query has one target (DOCCI); otherwise AP at the run's cutoff (AP@100, AP@50)."""
    single = all(r["nrel"] == 1 for r in rows)
    m = re.search(r"@(\d+)", rows[0]["metric"] or "")
    cap = int(m.group(1)) if m and not single else None

    def metric(score, row) -> float:
        o = np.argsort(-np.nan_to_num(np.asarray(score, float), nan=-1e9), kind="stable")
        y = row["y"]
        if single:
            return float(y[o[0]])
        o = o if cap is None else o[:cap]
        yy = y[o]
        den = min(row["nrel"], cap) if cap else row["nrel"]
        return float("nan") if den <= 0 else float((np.cumsum(yy) / np.arange(1, len(yy) + 1) * yy).sum() / den)
    return metric


def match_values(P: dict):
    """One column per question that bears on the match (not the descriptors F), 1 speaking for a match."""
    cols, V = [], []
    for x in P["qds"]:
        if x["family"] == "F":
            continue
        v = P["val"][x["question_id"]]
        cols.append(x)
        V.append(1 - v if x["family"] == "N" else v)
    return cols, fill(np.stack(V, 1))


def family_scores(P: dict) -> dict:
    cols, V = match_values(P)
    out = {}
    for f in FAM:
        j = [k for k, x in enumerate(cols) if x["family"] == f]
        if not j:
            continue
        if f == "R":
            req = [k for k in j if cols[k]["required"] is not False] or j
            out["R"] = geo(V[:, req])
        else:
            out[f] = V[:, j].mean(1)
    return out


def fusions(P: dict) -> dict:
    cols, V = match_values(P)
    fs = family_scores(P)
    Z = (V - V.mean(0)) / (V.std(0) + 1e-6)
    w = 1 - V.mean(0) + 1e-3
    return {"families": np.mean([fs[f] for f in FAM if f in fs], 0), "questions": V.mean(1), "z": mm(Z.mean(1)),
            "rarity": (V * w).sum(1) / w.sum()}


def earlier(row: dict) -> dict:
    """The earlier run's seven questions on the same photos: the whole query, and Eq. 2's fusion before the search."""
    o = row["old"]
    h, r = fill(o["h"])[:, 0], fill(o["r"])[:, 0]
    s = geo(o["S"], o["w"]) if o["S"].shape[1] else h
    return {"h": h, "pipe": 0.4 * h + 0.4 * s + 0.2 * r}


def with_prior(s, row: dict):
    return 0.8 * np.asarray(s, float) + 0.2 * mm(row["sim"])


def scores(row: dict, p: str) -> dict:
    """Every ranking of the comparison for one query and pass, each with the search score."""
    P = row["passes"][p]
    e = earlier(row)
    sc = {"earlier h": e["h"], "earlier": e["pipe"], **family_scores(P), **fusions(P)}
    sc["both"] = 0.5 * e["pipe"] + 0.5 * sc["families"]
    return {k: with_prior(v, row) for k, v in sc.items()} | {"search": row["sim"]}


def questions_per_photo(rows: list[dict], p: str = "plain") -> float:
    return float(np.mean([len(r["passes"][p]["qds"]) for r in rows if p in r["passes"]]))


def compare(source: str, subset=None) -> dict:
    """The larger question set against the seven questions on the same photos, each pass, every ranking with the search
    score: the mean metric of every ranking, the paired gain of the families' mean over the seven questions, and (with
    both passes) the brief's change to it."""
    rows = [r for r in load(source) if subset is None or subset(r)]
    metric = metric_of(rows)
    out = {"n": len(rows), "questions_per_photo": questions_per_photo(rows)}
    for p in ("plain", "brief"):
        if not all(p in r["passes"] for r in rows):
            continue
        per = defaultdict(list)
        for r in rows:
            for k, v in scores(r, p).items():
                per[k].append(metric(v, r))
        out[p] = {k: float(np.nanmean(v)) for k, v in per.items()}
        out[p + ".per"] = {k: np.array(v) for k, v in per.items()}
        out[p + ".gain"] = boot(np.array(per["families"]) - np.array(per["earlier"]))
    if "brief" in out:
        out["brief_minus_plain"] = boot(out["brief.per"]["families"] - out["plain.per"]["families"])
    return out
