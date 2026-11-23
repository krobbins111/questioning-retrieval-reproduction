"""Table 4's feedback to the search, on INQUIRE-Fullrank, from the saved SigLIP2 searches over the 5 million photos
(inquire_full_streams, inquire_full_steer) and the first stages' rankings (inquire_full_*_lc_*/firststage). Labels
only score, except in the control "with the labels".

rewording   the union of the top k of every saved search of a query, against the query's own ranking at the same
            number of photos: the share of the query's relevant photos found (k = 20 in Table 4)
llm_text    after the first 100 photos of the query's own ranking, the next 100 from the search with every LLM text and
            the query as one vector, against the own ranking going on
steer       after the first 100 photos of CLIP ViT-H's ranking are checked, the next 100 unseen photos of SigLIP2's
            search steered by the answers, against SigLIP2's search for the query (and the ViT-H ranking going on)

All follow the analysis scripts that printed the numbers, including their bootstrap: rewording draws every interval
from one generator (2,000 resamples, in the analysis's order of splits, groups of searches and k), llm_text and steer a
fresh generator per interval (4,000 resamples).
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq

from qr.data import data_root, load, relevant


@lru_cache(maxsize=2)
def _streams() -> dict:
    t = pq.read_table(data_root() / "inquire_full_streams/streams.parquet").to_pydict()
    S: dict = defaultdict(dict)
    pos = {}
    for q, p, n, ids in zip(t["query_id"], t["position"], t["stream"], t["ids"]):
        S[q][n] = ids
        pos[q] = p
    return {"streams": dict(S), "position": pos}


@lru_cache(maxsize=4)
def _first(source: str) -> dict:
    """{query: the first stage's top 1,000 image ids, in rank order}."""
    t = pq.read_table(data_root() / f"{source}/firststage.parquet", columns=["query_id", "rank", "image_id"]).to_pydict()
    o = defaultdict(list)
    for q, r, i in sorted(zip(t["query_id"], t["rank"], t["image_id"]), key=lambda x: (x[0], x[1])):
        o[q].append(i)
    return dict(o)


def _own(split: str) -> dict:
    return _first(f"inquire_full_siglip2_lc_{split}")


def _queries(split: str) -> list[str]:
    s = _streams()
    return sorted((q for q in s["streams"] if q.startswith(split + "_")), key=lambda q: s["position"][q])


def rewording(k_report: int = 20) -> dict:
    """{split: (found by the searches' union, found by the own ranking, difference, low, high)} at k = k_report."""
    rng = np.random.default_rng(0)

    def boot(dv):
        dv = np.asarray(dv, float)
        bs = [dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(2000)]
        return float(dv.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    S = _streams()["streams"]
    out = {}
    for split in ("val", "test"):
        Q, _ = load(f"inquire_full_siglip2_lc_{split}")
        own = _own(split)
        qs = _queries(split)
        for group in (None, "anchored", "alone"):
            for k in (10, 20, 50, 100, 200):
                fu, fo = [], []
                for q in qs:
                    r = relevant(Q[q])
                    names = [n for n in S[q] if group is None or (group == "anchored") == n.startswith("q+")]
                    U = set(own[q][:k])
                    for n in names:
                        U |= set(S[q][n][:k])
                    base = set(own[q][:min(len(U), 1000)])
                    fu.append(len(U & r) / max(1, len(r))); fo.append(len(base & r) / max(1, len(r)))
                m, lo, hi = boot(np.array(fu) - np.array(fo))
                if group is None and k == k_report:
                    out[split] = (float(np.mean(fu)), float(np.mean(fo)), m, lo, hi)
    return out


def llm_text(first: int = 100, add: int = 100) -> dict:
    """{split: (found with the query and every LLM text as one vector, found by the own ranking, difference, low, high)}."""
    from qr.bank import boot as boot4000
    S = _streams()["streams"]
    out = {}
    for split in ("val", "test"):
        Q, _ = load(f"inquire_full_siglip2_lc_{split}")
        own = _own(split)
        x, ref = [], []
        for q in _queries(split):
            P = relevant(Q[q]); npos = max(1, len(P)); base = own[q]; seen = set(base[:first])

            def nxt(stream, seen_, n):
                o = []
                for i in stream:
                    if len(o) >= n:
                        break
                    if i not in seen_:
                        o.append(i)
                return o

            def score(extra):
                extra = list(extra) + nxt(base, seen | set(extra), add - len(extra))
                return len(P & (seen | set(extra[:add]))) / npos
            o = score(base[first:first + add])
            ref.append(o)
            x.append(score(nxt(S[q]["q+all"], seen, add)) if "q+all" in S[q] else o)
        m, lo, hi = boot4000(np.array(x) - np.array(ref))
        out[split] = (float(np.mean(x)), float(np.mean(ref)), m, lo, hi)
    return out


STEER = "inquire_full_steer/searches.parquet"


def has_steer() -> bool:
    return (data_root() / STEER).exists()


@lru_cache(maxsize=1)
def _steer_searches() -> dict:
    t = pq.read_table(data_root() / STEER).to_pydict()
    S: dict = defaultdict(dict)
    pos = {}
    for sp, q, p, b, s, ids in zip(t["split"], t["query_id"], t["position"], t["checked"], t["search"], t["ids"]):
        S[(sp, q)][(b, s)] = ids
        pos[(sp, q)] = p
    return {"searches": dict(S), "position": pos}


def steer(checked: int = 100, ks: tuple = (100, 200, 500)) -> dict:
    """{split: {K: {vith, text, steered, gain (mean, low, high), labels, labels_gain, queries}}}: after the first
    `checked` photos of CLIP ViT-H's ranking, the share of all of a query's relevant photos among the next K unseen
    photos of the ViT-H ranking going on, of SigLIP2's search for the query, of that search steered by the answers, and
    steered with the labels (a query without that search keeps SigLIP2's). Queries without a relevant photo or without
    the steered search are left out, as in the analysis."""
    from qr.bank import boot as boot4000
    T = _steer_searches()
    out = {}
    for split in ("val", "test"):
        Q, _ = load(f"inquire_full_vith_lc_{split}")
        vith = _first(f"inquire_full_vith_lc_{split}")
        rows = defaultdict(list)
        for q in sorted((q for sp, q in T["searches"] if sp == split), key=lambda q: T["position"][(split, q)]):
            r = relevant(Q[q]); n = len(r); s = T["searches"][(split, q)]
            if n == 0 or (0, "text") not in s or (checked, "answers") not in s:
                continue
            seen = set(vith[q][:checked])

            def nxt(stream, K):
                o = k = 0
                for i in stream:
                    if i in seen:
                        continue
                    o += i in r; k += 1
                    if k >= K:
                        break
                return o
            for K in ks:
                rows[("vith", K)].append(nxt(vith[q], K) / n)
                rows[("text", K)].append(nxt(s[(0, "text")], K) / n)
                rows[("steered", K)].append(nxt(s[(checked, "answers")], K) / n)
                rows[("labels", K)].append(nxt(s.get((checked, "labels"), s[(0, "text")]), K) / n)
        out[split] = {}
        for K in ks:
            v, t, a, o = (np.array(rows[(k, K)]) for k in ("vith", "text", "steered", "labels"))
            out[split][K] = {"vith": float(v.mean()), "text": float(t.mean()), "steered": float(a.mean()), "gain": boot4000(a - t),
                             "labels": float(o.mean()), "labels_gain": float((o - t).mean()), "queries": len(t)}
    return out
