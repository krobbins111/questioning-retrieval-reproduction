"""The paper's scoring rule (Sec. 3, Eq. 1-2) on the released tables, and the lead used to decide (Sec. 5).

    S = exp( sum_j w_j log max(s_j, 0.02) / sum_j w_j )          weighted geometric mean of the five statements
    f = 0.8 (0.4 h + 0.4 S + 0.2 r) + 0.2 c                        c: search score, min-max scaled within the pool

h: P(yes) for the whole query; s_j, w_j: P(yes) and weight of statement j; r: expected 1-to-5 rating scaled to [0, 1].
Photos are ranked by f, ties broken by the search rank. A photo with a failed request keeps only 0.2 c, so it sinks but
keeps its search order. The lead is f of the top photo minus f of the second.

`weights` and `alpha` give the ablations of the paper (whole query alone, whole query with the search score, ...): a
signal with weight 0 still counts as present, as in the code that produced the paper. `mode="combsum_z"` is the
fusion without weights of Sec. 3: each answer as log-odds, standardised over the pool, summed with the standardised
search score (the five statements are averaged into one column first).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

W_HOLISTIC, W_SUB, W_RELEVANCE, ALPHA, FLOOR = 0.4, 0.4, 0.2, 0.2, 0.02
WEIGHTS = (W_HOLISTIC, W_SUB, W_RELEVANCE)


@dataclass
class Photo:
    image_id: str
    search_rank: int
    search_score: float
    h: float | None                                   # whole query, P(yes)
    s: list = field(default_factory=list)             # statements, P(yes), in question order; None if not answered
    w: list = field(default_factory=list)             # statement weights, same order
    r: float | None = None                            # rating, expected value in [0, 1]
    error: bool = False


def statements(s: list, w: list) -> float | None:
    pw = [(sj, wj) for sj, wj in zip(s, w) if sj is not None]
    if not pw:
        return None
    return math.exp(sum(wj * math.log(min(1.0, max(sj, FLOOR))) for sj, wj in pw) / sum(wj for _, wj in pw))


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [0.5] * len(xs) if hi - lo < 1e-9 else [(x - lo) / (hi - lo) for x in xs]


def _logit(p: float, floor: float = FLOOR) -> float:
    p = min(1.0 - floor, max(floor, p))
    return math.log(p / (1.0 - p))


def _z(xs: list) -> list[float]:
    """Standardise over the pool; missing values (failed photos) get the pool minimum."""
    vals = [x for x in xs if x is not None]
    if len(vals) < 2:
        return [0.0] * len(xs)
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
    if sd < 1e-9:
        return [0.0] * len(xs)
    zs = [(x - mu) / sd if x is not None else None for x in xs]
    lo = min(z for z in zs if z is not None)
    return [z if z is not None else lo for z in zs]


def scores(photos: list[Photo], alpha: float = ALPHA, weights: tuple = WEIGHTS, mode: str = "weighted") -> list[float]:
    """f for every photo, in the order given."""
    if mode == "combsum_z":
        ok = [not p.error for p in photos]
        hol = [_logit(p.h) if p.h is not None and o else None for p, o in zip(photos, ok)]
        rel = [_logit(p.r) if p.r is not None and o else None for p, o in zip(photos, ok)]
        cols = []
        for j in range(max((len(p.s) for p in photos), default=0)):
            col = [_logit(p.s[j]) if o and j < len(p.s) and p.s[j] is not None else None for p, o in zip(photos, ok)]
            if any(v is not None for v in col):
                cols.append(_z(col))
        zsub = [sum(c[i] for c in cols) / len(cols) for i in range(len(photos))] if cols else [0.0] * len(photos)
        zh, zr, zs = _z(hol), _z(rel), _z([p.search_score for p in photos])
        return [zh[i] + zsub[i] + zr[i] + zs[i] for i in range(len(photos))]
    wh, ws, wr = weights
    c = _minmax([p.search_score for p in photos])
    out = []
    for p, ci in zip(photos, c):
        parts = [(v, wt) for v, wt in ((p.h, wh), (statements(p.s, p.w), ws), (p.r, wr)) if v is not None]
        if p.error or not parts:
            out.append(alpha * ci)
            continue
        d = sum(v * wt for v, wt in parts) / (sum(wt for _, wt in parts) or (wh + ws + wr))
        out.append((1 - alpha) * d + alpha * ci)
    return out


def rank(photos: list[Photo], alpha: float = ALPHA, weights: tuple = WEIGHTS, mode: str = "weighted") -> list[tuple[Photo, float]]:
    """Return (photo, f) best first; ties go to the better search rank."""
    out = list(zip(photos, scores(photos, alpha, weights, mode)))
    out.sort(key=lambda x: (-round(x[1], 9), x[0].search_rank))
    return out


def lead(ranked: list[tuple[Photo, float]]) -> float:
    return ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")


def passes_all(p: Photo) -> bool:
    """Yes (P >= 0.5) to the whole query and to every statement; an unanswered statement counts as no."""
    return p.h is not None and p.h >= 0.5 and all((x if x is not None else 0.0) >= 0.5 for x in p.s)
