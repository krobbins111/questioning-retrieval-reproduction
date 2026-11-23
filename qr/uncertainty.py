"""Table 4's "report uncertainty" row: can the verifier say that it cannot judge a photo? The larger question set
(qr/bank.py) was asked of every photo checked for an INQUIRE-Fullrank query (bank_inquire_wide_*), plainly and with
the LLM's brief. Its question on the main problem with the photo (X) offers "The image is too unclear, small or cropped
to tell", and its detail (C) and descriptor (F) questions offer "Cannot tell from this image".

On CLIP ViT-H's top 500, ranked by the seven questions with the search score the run stored (Eq. 2; SigLIP2's
similarity, score_vith), photos are grouped by label and by rank, as the analysis that printed the paper's numbers
grouped them:

  chosen     how often "too unclear" is the likeliest answer to X: relevant photos at ranks 1-50, 51-200 and 201-500,
             photos that are not relevant at 1-50 and at 51-500
  evidence   how often the highest probability of "cannot tell" (C, F), of "too unclear" (X), or of any of them is at
             least 0.3: relevant at 1-50, relevant at 201-500 (wrongly rejected), not relevant at 1-50 (wrongly
             accepted), not relevant at 201-500 (correctly rejected)

Shares pool the photos of all queries. Labels only group the photos.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq

from qr import bank as QB
from qr.data import data_root

CANNOT = "Cannot tell from this image"
UNCLEAR = 3                                       # X's options: another subject, a detail missing, nothing wrong, too unclear
PASSES = ("plain", "brief")
CHOSEN = {"relevant 1-50": lambda y, rk: y & (rk <= 50), "relevant 51-200": lambda y, rk: y & (rk > 50) & (rk <= 200),
          "relevant 201-500": lambda y, rk: y & (rk > 200), "not relevant 1-50": lambda y, rk: ~y & (rk <= 50),
          "not relevant 51-500": lambda y, rk: ~y & (rk > 50)}
EVIDENCE = {"relevant 1-50": lambda y, rk: y & (rk <= 50), "relevant 201-500": lambda y, rk: y & (rk > 200),
            "not relevant 1-50": lambda y, rk: ~y & (rk <= 50), "not relevant 201-500": lambda y, rk: ~y & (rk > 200)}
PHOTO_COLS = ["query_id", "rank", "image_id", "relevant", "earlier_h", "earlier_r", "earlier_s", "rank_vith", "score_vith"]


def queries(source: str):
    """One query at a time (the answers keep one row group per query): ids, labels, the ViT-H pool's ranks and search
    scores, the earlier run's answers, and per pass the questions and every multiple-choice question's probabilities."""
    d = data_root() / source
    Q = {r["query_id"]: r for r in pq.read_table(d / "queries.parquet").to_pylist()}
    ph = pq.read_table(d / "photos.parquet", columns=PHOTO_COLS).to_pydict()
    at = defaultdict(list)
    for k, q in enumerate(ph["query_id"]):
        at[q].append(k)
    K = defaultdict(lambda: defaultdict(list))
    for r in pq.read_table(d / "questions.parquet").to_pylist():
        K[r["query_id"]][r["pass"]].append(r)
    nan = lambda x: np.nan if x is None else x  # noqa: E731

    def build(q, ans):
        rows = sorted(at[q], key=lambda k: ph["rank"][k])
        ids = [ph["image_id"][k] for k in rows]
        passes = {}
        for p, qds in K[q].items():
            prob = {x["question_id"]: np.array([ans[(p, x["question_id"])].get(i) or [np.nan] * len(x["options"] or x["levels"] or [])
                                                for i in ids], float) for x in qds if x["kind"] != "noul"}
            passes[p] = {"qds": qds, "prob": prob}
        return {"query_id": q, "ids": ids, "y": np.array([ph["relevant"][k] for k in rows], bool), "nrel": Q[q]["n_relevant"],
                "rk_vith": np.array([10 ** 6 if ph["rank_vith"][k] is None else ph["rank_vith"][k] for k in rows]),
                "sim_vith": np.array([0.0 if ph["score_vith"][k] is None else ph["score_vith"][k] for k in rows], float),
                "old": {"h": np.array([nan(ph["earlier_h"][k]) for k in rows], float), "r": np.array([nan(ph["earlier_r"][k]) for k in rows], float),
                        "S": np.array([[nan(x) for x in (ph["earlier_s"][k] or [])] for k in rows], float).reshape(len(rows), -1),
                        "w": np.array(Q[q]["earlier_w"] or [], float)},
                "passes": passes}
    f = pq.ParquetFile(d / "answers.parquet")
    cur, ans = None, None
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=["query_id", "pass", "image_id", "question_id", "probs"]).to_pydict()
        for q, p, i, k, pr in zip(t["query_id"], t["pass"], t["image_id"], t["question_id"], t["probs"]):
            if q != cur:
                if cur is not None:
                    yield build(cur, ans)
                cur, ans = q, defaultdict(dict)
            ans[(p, k)][i] = pr
    if cur is not None:
        yield build(cur, ans)


def _cannot(P: dict, family: str, m) -> np.ndarray:
    """The highest probability of "cannot tell" among the family's multiple-choice questions that offer it (0 if none)."""
    out = [np.nan_to_num(P["prob"][x["question_id"]], nan=0.0)[m][:, x["options"].index(CANNOT)]
           for x in P["qds"] if x["family"] == family and x["kind"] == "choice" and CANNOT in (x["options"] or [])
           and x["question_id"] in P["prob"]]
    return np.max(out, 0) if out else np.zeros(int(m.sum()))


@lru_cache(maxsize=2)
def report(split: str) -> dict:
    """{"chosen": {pass: {group: share}}, "evidence": {pass: {group: {"C", "F", "X", "any": share}}},
    "n": {group: photos}, "queries": n}, shares from 0 to 1."""
    cnt = {p: {g: np.zeros(4) for g in CHOSEN} for p in PASSES}
    ev = {p: {g: defaultdict(list) for g in EVIDENCE} for p in PASSES}
    n, nq = defaultdict(int), 0
    for r in queries(f"bank_inquire_wide_{split}"):
        nq += 1
        m = r["rk_vith"] <= 500
        y = r["y"][m]
        f = 0.8 * np.nan_to_num(QB.earlier(r)["pipe"], nan=0.0)[m] + 0.2 * QB.mm(r["sim_vith"][m])
        rk = np.empty(len(f), int)
        rk[np.argsort(-f, kind="stable")] = np.arange(1, len(f) + 1)
        for p in PASSES:
            P = r["passes"][p]
            x = next(q for q in P["qds"] if q["question_id"] == "x")
            assert x["family"] == "X" and x["options"][UNCLEAR].startswith("The image is too unclear"), r["query_id"]
            X = np.nan_to_num(P["prob"]["x"], nan=0.0)[m]
            a = X.argmax(1)
            for g, fn in CHOSEN.items():
                cnt[p][g] += np.bincount(a[fn(y, rk)], minlength=4)[:4]
            c, ff, xu = _cannot(P, "C", m), _cannot(P, "F", m), X[:, UNCLEAR]
            u = {"C": c, "F": ff, "X": xu, "any": np.maximum(np.maximum(c, ff), xu)}
            for g, fn in EVIDENCE.items():
                sel = fn(y, rk)
                for k, v in u.items():
                    ev[p][g][k].append(v[sel])
                if p == PASSES[0]:
                    n[g] += int(sel.sum())
    return {"chosen": {p: {g: float(c[UNCLEAR] / max(c.sum(), 1)) for g, c in cnt[p].items()} for p in PASSES},
            "evidence": {p: {g: {k: float(np.mean(np.concatenate(v) >= 0.3)) for k, v in ev[p][g].items()} for g in EVIDENCE} for p in PASSES},
            "n": dict(n), "queries": nq}


def summary(split: str) -> dict:
    """Table 4's numbers, in percent: the lowest and highest share of photos whose likeliest answer is "too unclear"
    (over the groups and both passes), and the share with any can't-judge evidence among wrongly rejected matches and
    among correctly rejected photos (asked plainly, with the brief)."""
    R = report(split)
    ch = [100 * v for p in PASSES for v in R["chosen"][p].values()]
    ev = lambda g: tuple(100 * R["evidence"][p][g]["any"] for p in PASSES)  # noqa: E731
    return {"chosen": (min(ch), max(ch)), "wrongly_rejected": ev("relevant 201-500"), "correctly_rejected": ev("not relevant 201-500")}
