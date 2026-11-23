#!/usr/bin/env python
"""Table 4's "steer the search": once the first 100 (or 500) photos of CLIP ViT-H's ranking for an INQUIRE-Fullrank
query have been checked, a SigLIP2 search over the 5 million photos steered by the verifier's answers, against the
plain SigLIP2 search for the query. Written to the data folder as inquire_full_steer/searches.parquet, one row per
query and search, with the search's top 1,000 photos:

  position  the query's place in the analysis's order (its bootstrap draws the queries in that order)
  checked   how many photos of ViT-H's ranking were checked before the search (100, 500; 0 for the plain search)
  search    "text": SigLIP2's search for the query (lowercased), the same whatever was checked
            "answers": the query's vector moved along a direction fitted on the checked photos' SigLIP2 vectors
            against their score from the seven questions (Eq. 2 without the search score)
            "labels": the same direction fitted against the labels instead, the control "with the labels"

The direction d solves ridge regression (penalty 1) in its dual form, d = A' (A A' + I)^-1 (t - mean t), with A the
checked photos' unit vectors centred and t their scores; the query's unit vector q becomes unit(q + sd(Xq)/sd(Xd) d),
a step as wide as the spread of the query's own similarities over the checked photos X. Unanswered statements count
0.5. A query whose labels are all equal among the checked photos has no "labels" search (the analysis used "text").

Inputs: the ViT-H runs' audits (answers), their results (labels), the first stage's ranking, SigLIP2's text vectors
for the lowercased queries, SigLIP2's image vectors (the explore pool's file first, then the 49 shards). The scan of
the shards takes a few minutes and can be split with --budget-seconds (run again to continue).

    python export/export_steer.py --jevrank <jevrank> --inquire-full <inquire_full> --out <data> --state <scratch folder>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from export_extras import write

SPLITS = ("val", "test")
CHECKED = (100, 500)
KEEP, SHARDS, CHUNK, TOP = 1700, 49, 250, 1000


def unit(x):
    x = np.asarray(x, np.float64)
    return x / max(np.linalg.norm(x), 1e-12)


def geo(P, w):
    P = np.clip(np.nan_to_num(P, nan=.5), .02, 1)
    return np.exp((np.log(P) * w).sum(1) / w.sum())


def ridge_dir(Fm, t, q, ridge=1.0):
    A = Fm - Fm.mean(0)
    d = A.T @ np.linalg.solve(A @ A.T + ridge * np.eye(len(A)), t - t.mean())
    nd = np.linalg.norm(d)
    if not np.isfinite(nd) or nd < 1e-9:
        return None, 0.0
    d = d / nd
    sq, sd = (Fm @ q).std(), (Fm @ d).std()
    return (d, sq / sd) if sq > 1e-9 and sd > 1e-9 else (None, 0.0)


def queries(J: Path, F: Path, split: str) -> list[dict]:
    """The analysis's queries in its order (the results' order): the checked photos of the ViT-H run in the order of
    its answers, with the seven answers (float32, as the analysis stored them), labels and first-stage ranks."""
    fs1 = json.load(open(F / "firststage/vith_lc.json", encoding="utf-8"))["queries"]
    res = json.load(open(J / f"results/inquire_full/full_vith_lc_{split}_k500.json", encoding="utf-8"))
    out = []
    for r in res["per_query"]:
        qid = r["qid"]; num = qid.split("_")[1]
        a = json.load(open(next((F / f"audit/full_vith_lc_{split}_k500").glob(f"*_{qid}.json")), encoding="utf-8"))
        qs = [q["id"] for q in a["questions"]]; w = {q["id"]: q.get("weight", 1.0) for q in a["questions"]}
        A = {}
        for x in a["responses"]:
            an = x.get("answers") or {}
            if an:
                A.setdefault(str(x["image_id"]), [((an.get(k) or {}).get("prob") if k != "relevance" else (an.get(k) or {}).get("expected")) for k in qs])
        o1 = fs1.get(num) or fs1.get(str(int(num)))
        r1 = {str(i): k for k, i in enumerate(o1["ids"])}
        rel = {str(x) for x in r["relevant"]}
        ids = list(A)
        out.append({"qid": qid, "num": int(num), "ids": ids, "qs": qs, "w": [w[k] for k in qs],
                    "M": np.array([[np.nan if t is None else t for t in A[i]] for i in ids], np.float32),
                    "y": np.array([i in rel for i in ids]), "nrel": len(rel), "rank_vith": np.array([r1.get(i, 99999) for i in ids])})
    return out


def vectors(F: Path, need: np.ndarray) -> np.ndarray:
    """SigLIP2 image vectors (float16) of the photos, from the explore pool's file where it has them, else the shards."""
    z = np.load(F / "xplore/inquire/pool_vecs.npz"); pid = z["ids"]; o = np.argsort(pid)
    pos = np.searchsorted(pid[o], need); pos[pos >= len(pid)] = 0; hit = pid[o][pos] == need
    X = np.zeros((len(need), 1152), np.float16); PX = z["X"]; X[hit] = PX[o[pos[hit]]]; del PX
    miss, mi = need[~hit], np.nonzero(~hit)[0]
    for s in range(SHARDS):
        if not len(miss):
            break
        ids = np.load(F / f"siglip2/ids_{s:03d}.npy"); oo = np.argsort(ids)
        p = np.searchsorted(ids[oo], miss); p[p >= len(ids)] = 0; h = ids[oo][p] == miss
        if not h.any():
            continue
        rows = oo[p[h]]; order = np.argsort(rows); E = np.load(F / f"siglip2/emb_{s:03d}.npy", mmap_mode="r")
        X[mi[h][order]] = np.asarray(E[rows[order]])
    assert not (np.abs(X.astype(np.float32)).sum(1) == 0).any(), "a photo without a SigLIP2 vector"
    return X


def prepare(J: Path, F: Path, state: Path) -> None:
    T = np.load(F / "siglip2/texts_lc.npz"); qtext = {int(i): v for i, v in zip(T["query_id"], T["query_emb"])}
    Q = {s: queries(J, F, s) for s in SPLITS}
    need = np.unique(np.array([int(i) for s in SPLITS for q in Q[s] for i, rk in zip(q["ids"], q["rank_vith"]) if rk < max(CHECKED)], np.int64))
    vid, VX = need, vectors(F, need)
    cols, names, meta = [], [], []
    for s in SPLITS:
        for qi, q in enumerate(Q[s]):
            qv = unit(qtext[q["num"]])
            ids = np.array([int(i) for i in q["ids"]]); keep = q["rank_vith"] < max(CHECKED)
            X = np.zeros((len(ids), 1152)); X[keep] = VX[np.searchsorted(vid, ids[keep])].astype(np.float64)
            X[keep] /= np.maximum(np.linalg.norm(X[keep], axis=1, keepdims=True), 1e-12)
            M, qs = q["M"], q["qs"]
            h = np.nan_to_num(M[:, qs.index("holistic")], nan=.5); r = np.nan_to_num(M[:, qs.index("relevance")], nan=.5)
            sj = [j for j, k in enumerate(qs) if k.startswith("sub")]
            sc_ = geo(M[:, sj], np.array([q["w"][j] for j in sj])) if sj else h
            jev = .4 * h + .4 * sc_ + .2 * r; y = q["y"].astype(float)
            cols.append(qv); names.append((s, q["qid"], qi, 0, "text"))
            for B in CHECKED:
                m = q["rank_vith"] < B
                if m.sum() < 20:
                    continue
                for tag, t in (("answers", jev), ("labels", y)):
                    d, sc = ridge_dir(X[m], t[m], qv)
                    if d is not None:
                        cols.append(unit(qv + 1.0 * sc * d)); names.append((s, q["qid"], qi, B, tag))
    W = np.stack(cols).astype(np.float32)
    np.savez(state / "steer_prep.npz", W=W, names=np.array(json.dumps(names)))
    print(f"prepared {len(W)} searches for {sum(len(Q[s]) for s in SPLITS)} queries ({len(need):,} photos' vectors)", flush=True)


def scan(F: Path, state: Path, budget: float) -> bool:
    """The shards, the top KEEP photos of every search kept as the analysis kept them. True when all are done."""
    W = np.load(state / "steer_prep.npz")["W"]; st = state / "steer_scan.npz"; t0 = time.time()
    if st.exists():
        z = np.load(st); top_s, top_i, start = z["s"], z["i"], int(z["next"])
    else:
        top_s = np.full((KEEP, len(W)), -np.inf, np.float32); top_i = np.zeros((KEEP, len(W)), np.int64); start = 0
    for sh in range(start, SHARDS):
        if budget and time.time() - t0 > budget:
            np.savez(st, s=top_s, i=top_i, next=sh); print("paused before shard", sh, flush=True)
            return False
        E = np.load(F / f"siglip2/emb_{sh:03d}.npy").astype(np.float32); gid = np.load(F / f"siglip2/ids_{sh:03d}.npy")
        for c0 in range(0, len(W), CHUNK):
            S = E @ W[c0:c0 + CHUNK].T
            np.negative(S, out=S)
            part = np.argpartition(S, KEEP - 1, axis=0)[:KEEP] if S.shape[0] > KEEP else np.broadcast_to(np.arange(S.shape[0])[:, None], S.shape)
            cs = -np.take_along_axis(S, part, 0); ci = gid[part]; del S
            alls = np.concatenate([top_s[:, c0:c0 + CHUNK], cs]); alli = np.concatenate([top_i[:, c0:c0 + CHUNK], ci])
            sel = np.argpartition(-alls, KEEP - 1, axis=0)[:KEEP]
            top_s[:, c0:c0 + CHUNK] = np.take_along_axis(alls, sel, 0); top_i[:, c0:c0 + CHUNK] = np.take_along_axis(alli, sel, 0)
        del E
    np.savez(st, s=top_s, i=top_i, next=SHARDS)
    print("scanned %d shards in %.0fs" % (SHARDS - start, time.time() - t0), flush=True)
    return True


def finish(state: Path, out: Path) -> list[dict]:
    names = json.loads(str(np.load(state / "steer_prep.npz")["names"]))
    z = np.load(state / "steer_scan.npz"); top_s, top_i = z["s"], z["i"]
    o = np.argsort(-top_s, axis=0, kind="stable"); top_i = np.take_along_axis(top_i, o, 0)
    rows = [dict(split=s, query_id=qid, position=qi, checked=B, search=tag, ids=[str(i) for i in top_i[:TOP, j]])
            for j, (s, qid, qi, B, tag) in enumerate(names)]
    S = pa.string()
    m = [write(rows, pa.schema([("split", S), ("query_id", S), ("position", pa.int32()), ("checked", pa.int32()), ("search", S),
                                ("ids", pa.list_(S))]), out / "inquire_full_steer/searches.parquet", out)]
    print(f"wrote {len(rows)} searches", flush=True)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jevrank", required=True, type=Path)
    ap.add_argument("--inquire-full", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", required=True, type=Path, help="a scratch folder for the steps between runs")
    ap.add_argument("--budget-seconds", type=float, default=0.0)
    a = ap.parse_args()
    a.state.mkdir(parents=True, exist_ok=True)
    if not (a.state / "steer_prep.npz").exists():
        prepare(a.jevrank, a.inquire_full, a.state)
        if a.budget_seconds:
            return
    if not scan(a.inquire_full, a.state, a.budget_seconds):
        return
    entry = finish(a.state, a.out)
    mf = a.out / "manifest.json"
    manifest = json.load(open(mf, encoding="utf-8"))
    manifest["inquire_full_steer"] = entry
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    for p in ("steer_prep.npz", "steer_scan.npz"):          # a later run starts afresh
        (a.state / p).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
