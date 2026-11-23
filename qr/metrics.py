"""Metrics and protocols of the paper, written to match the scripts that produced its numbers.

ap_at      INQUIRE's AP@k: sklearn's AP over the top k rescaled by (#relevant in the top k) / min(#relevant, k), i.e.
           the sum of precision at each hit over min(#relevant, k). k = 100 for INQUIRE-Rerank, 50 for Fullrank.
tau_for    the threshold of Sec. 5: the lowest lead whose answers above it are right at least `target` of the time
           (at least nmin answers above it).
halves     the 40 random halves of Table 2, each used both ways (seed s: numpy default_rng(s).permutation).
boot_ci    paired bootstrap over queries: 10,000 resamples from default_rng(0), 2.5 and 97.5 percentiles.
"""
from __future__ import annotations

import numpy as np


def ap_at(ids: list[str], rel: set[str], k: int) -> float:
    hits = s = 0.0
    for i, x in enumerate(ids[:k], 1):
        if x in rel:
            hits += 1
            s += hits / i
    return s / max(1, min(len(rel), k))


def auroc(v, ok) -> float:
    v, ok = np.asarray(v, float), np.asarray(ok, bool)
    pos, neg = v[ok], v[~ok]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def tau_for(v, ok, target: float = 0.98, nmin: int = 20) -> float:
    v, ok = np.asarray(v, float), np.asarray(ok, bool)
    order = np.argsort(-v)
    cum = np.cumsum(ok[order]) / np.arange(1, len(order) + 1)
    m = max([i + 1 for i in range(len(order)) if cum[i] >= target and i >= nmin] or [0])
    return float(v[order[m - 1]]) if m else float("inf")


def halves(n: int, splits: int = 40):
    for seed in range(splits):
        perm = np.random.default_rng(seed).permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        yield a, b
        yield b, a


def boot_ci(d, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def sure_protocol(lead: dict, ok: dict, levels: list[str], target: float = 0.98) -> dict:
    """Table 2: threshold from half of the targets (all their levels pooled), sure share and right-when-sure on the
    other half, averaged over 40 random halves used both ways. lead[l], ok[l]: arrays over the same targets."""
    n = len(ok[levels[0]])
    share, prec, ps, po, taus = {l: [] for l in levels}, {l: [] for l in levels}, [], [], []
    for tune, test in halves(n):
        tau = tau_for(np.concatenate([lead[l][tune] for l in levels]), np.concatenate([ok[l][tune] for l in levels]), target)
        taus.append(tau)
        ss, oo = [], []
        for l in levels:
            s = lead[l][test] >= tau
            share[l].append(s.mean())
            prec[l].append(ok[l][test][s].mean() if s.any() else np.nan)
            ss.append(s)
            oo.append(ok[l][test])
        ss, oo = np.concatenate(ss), np.concatenate(oo)
        ps.append(ss.mean())
        po.append(oo[ss].mean() if ss.any() else np.nan)
    return {"sure": {l: float(np.mean(share[l])) for l in levels}, "right": {l: float(np.nanmean(prec[l])) for l in levels},
            "sure_pooled": float(np.mean(ps)), "right_pooled": float(np.nanmean(po)), "taus": taus}


def ask_until_sure(lead: dict, ok: dict, used: dict, levels: list[str], target: float = 0.98) -> dict:
    """Table 3: on the targets of the other half, answer at the first level whose lead reaches the threshold, else at
    the last level; mean exact match and mean sentences used over the 80 runs."""
    n = len(ok[levels[0]])
    acc, nu = [], []
    for tune, test in halves(n):
        tau = tau_for(np.concatenate([lead[l][tune] for l in levels]), np.concatenate([ok[l][tune] for l in levels]), target)
        done = np.zeros(len(test), bool)
        right = np.zeros(len(test), bool)
        u = np.zeros(len(test))
        for l in levels:
            s = ((lead[l][test] >= tau) | (l == levels[-1])) & ~done
            right[s] = ok[l][test][s]
            u[s] = used[l][test][s]
            done |= s
        acc.append(right.mean())
        nu.append(u.mean())
    return {"exact": float(np.mean(acc)), "sentences": float(np.mean(nu))}
