#!/usr/bin/env python
"""Every number of the paper that the released tables determine, recomputed from the tables alone and set against the
value printed in the submitted paper. No verifier calls, no images, no embeddings.

    python paper/paper_numbers.py                    # every group
    python paper/paper_numbers.py table1 table2      # some groups: table1 sec4 depth table2 sec5 table3 sec6 sec7 figures examples clef fusion

One line per number: computed value, the paper's value, and `ok` when the computed value rounds to the printed one.
Writes results/numbers.json ({key: value}); paper/build.py typesets the tables from it.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qr.data import data_root, load, meta, relevant           # noqa: E402
from qr.metrics import ap_at, auroc, boot_ci, sure_protocol, tau_for, halves  # noqa: E402
from qr.score import lead, passes_all, rank                    # noqa: E402
from qr.gpt4o import Nov14Stream, gpt4o_ap                     # noqa: E402

LEVELS = ("s1", "s2", "s3", "s4", "all")
KS = (100, 200, 500)
RULES = {  # name: (weights of whole query, statements, 1-to-5 score; search weight alpha)
    "paper": ((0.4, 0.4, 0.2), 0.2),
    "no_search": ((0.4, 0.4, 0.2), 0.0),
    "whole": ((1.0, 0.0, 0.0), 0.0),
    "whole_search": ((1.0, 0.0, 0.0), 0.2),
    "whole_rating_search": ((0.4, 0.0, 0.2), 0.2),
    "statements_search": ((0.0, 1.0, 0.0), 0.2),
}
OUT: dict[str, float] = {}
ROWS: list[tuple] = []


SUPP = ("depth.", "fusion.", "clef.docci.s1", "examples.", "limits.")      # values of the draft supplement tables, not printed in the paper


def put(key: str, value: float, paper: float | None, nd: int = 1, label: str = "") -> None:
    """paper: the value as printed (None: computed for the supplement only). Keys starting with SUPP are checked against
    the draft supplement tables rather than the submitted paper."""
    OUT[key] = value
    ok = None if paper is None else round(value + 1e-12, nd) == round(paper, nd)
    ROWS.append((key, label, value, paper, nd, ok))
    mark = "" if ok is None else ("ok" if ok else "DIFF")
    d = max(nd, 0)
    src = "supp" if key.startswith(SUPP) and paper is not None else "paper"
    print(f"  {label or key:<68} {value:>9.{d}f}   {src} {'' if paper is None else f'{paper:.{d}f}':>7}   {mark}", flush=True)


def ranked_ids(photos, rule: str = "paper", mode: str = "weighted") -> list[str]:
    w, a = RULES[rule]
    return [p.image_id for p, _ in rank(photos, a, w, mode)]


def search_ids(photos) -> list[str]:
    """The search's own order: by its score (on INQUIRE-Rerank, SigLIP2 with the lowercased query), then its rank."""
    return [p.image_id for p in sorted(photos, key=lambda p: (-p.search_score, p.search_rank))]


# ---------------------------------------------------------------------------------------------------------- Table 1
def rerank_ap(source: str, order, subset=None) -> float:
    Q, P = load(source)
    aps = [ap_at(order(P[q]), relevant(Q[q]), 100) for q in P if subset is None or subset(Q[q])]
    return 100 * float(np.mean(aps))


def fullrank_ap(source: str, k: int | None, rule: str | None) -> np.ndarray:
    """Per-query AP@50 (official mean over every query), the verifier checking the search's top k."""
    Q, P = load(source)
    out = []
    for q in sorted(P):
        ph = [p for p in P[q] if k is None or p.search_rank <= k]
        out.append(ap_at(search_ids(ph) if rule is None else ranked_ids(ph, rule), relevant(Q[q]), 50))
    return np.array(out)


def gpt4o_paired(depths=(100,)) -> dict | None:
    """The paired difference with GPT-4o's published reranking (Sec. 4), on the test queries with its answers, at each
    depth in the ViT-H pool, with the bootstrap of the analysis that printed it (qr.gpt4o.Nov14Stream). None without
    GPT-4o's answers (scripts/gpt4o_answers.py)."""
    G = gpt4o_ap("test")
    if G is None:
        return None
    src = "inquire_full_vith_lc_test"
    Q, P = load(src)
    qs = sorted(P)
    sig = {(q, i): s for q, i, s in zip(*pq.read_table(data_root() / src / "candidates.parquet",
                                                       columns=["query_id", "image_id", "siglip2_score"]).to_pydict().values())}
    paired = np.array([q in G for q in qs])
    gpt = np.array([G[q] for q in qs if q in G])
    stream, out = Nov14Stream(), {}
    for k in depths:
        own = fullrank_ap(src, k, "paper")
        with_sig = np.array([ap_at(ranked_ids([replace(p, search_score=sig[(q, p.image_id)]) for p in P[q] if p.search_rank <= k]),
                                   relevant(Q[q]), 50) for q in qs])
        m, lo, hi = stream.depth(own, with_sig, gpt, paired)
        d = own[paired] - gpt
        out[k] = dict(n=int(paired.sum()), jev=float(own[paired].mean()), gpt4o=float(gpt.mean()), diff=m, lo=lo, hi=hi,
                      better=int((d > 0).sum()), worse=int((d < 0).sum()))
    out["stream"] = stream
    return out


def docci_top1(source: str, rule: str | None = "paper", answers_dir: str | None = None) -> float:
    Q, P = load(source, answers_dir=answers_dir)
    hit = [(search_ids(P[q]) if rule is None else ranked_ids(P[q], rule))[0] in relevant(Q[q]) for q in P]
    return 100 * float(np.mean(hit))


def winoground_scores(mode: str = "single") -> dict:
    rows = [r for r in pq.read_table(data_root() / "winoground/answers.parquet").to_pylist() if r["mode"] == mode]
    by = defaultdict(dict)
    for r in rows:
        by[r["example"]][(r["caption"], r["image"])] = r["p_yes"]
    t = i = g = 0
    margins, groups = [], []
    for p in by.values():
        tt = p[(0, 0)] > p[(1, 0)] and p[(1, 1)] > p[(0, 1)]
        ii = p[(0, 0)] > p[(0, 1)] and p[(1, 1)] > p[(1, 0)]
        t, i, g = t + tt, i + ii, g + (tt and ii)
        # how sure, not whether right: the smallest absolute difference of the four comparisons (wino_lead.py)
        margins.append(min(abs(p[(0, 0)] - p[(1, 0)]), abs(p[(1, 1)] - p[(0, 1)]), abs(p[(0, 0)] - p[(0, 1)]), abs(p[(1, 1)] - p[(1, 0)])))
        groups.append(tt and ii)
    n = len(by)
    return {"text": 100 * t / n, "image": 100 * i / n, "group": 100 * g / n, "auroc_min_margin": auroc(margins, groups), "n": n}


def table1() -> None:
    print("Table 1. Ranking against published numbers")
    dc = lambda q: meta(q).get("supercategory") != "Species"     # noqa: E731  Divide and Conquer's 154 queries
    put("t1.rerank.search", rerank_ap("inquire_rerank_test", search_ids), 53.6, label="INQUIRE-Rerank, search alone (AP)")
    put("t1.rerank.verifier", rerank_ap("inquire_rerank_test", ranked_ids), 63.9, label="INQUIRE-Rerank, with the verifier")
    Q, P = load("inquire_rerank_test")
    put("t1.dc.n", float(sum(dc(Q[q]) for q in P)), 154, 0, label="Divide and Conquer's queries (outside Species)")
    put("t1.dc.search", rerank_ap("inquire_rerank_test", search_ids, dc), 54.1, label="  their search alone")
    put("t1.dc.verifier", rerank_ap("inquire_rerank_test", ranked_ids, dc), 64.8, label="  their verifier")
    paper = {"vith_lc": {"search": 33.4, "paper": (46.2, 48.6, 49.7), "whole": (43.7, 44.1, 42.5), "whole_search": (45.0, 46.8, 47.3)},
             "siglip2_lc": {"search": 38.3, "paper": (46.7, 48.0, 49.0)}}
    for pool, pp in paper.items():
        src = f"inquire_full_{pool}_test"
        put(f"t1.full.{pool}.search", 100 * fullrank_ap(src, None, None).mean(), pp["search"], label=f"INQUIRE-Fullrank {pool}, search alone (mAP@50)")
        for rule in ("paper", "whole", "whole_search"):
            for k, v in zip(KS, pp.get(rule, (None,) * 3)):
                put(f"t1.full.{pool}.{rule}.{k}", 100 * fullrank_ap(src, k, rule).mean(), v, label=f"  {rule}, {k} checked")
    put("t1.docci.search", docci_top1("docci_all", None), 75.4, label="DOCCI all sentences, search alone (exact first, %)")
    put("t1.docci.verifier", docci_top1("docci_all"), 98.4, label="DOCCI all sentences, with the verifier")
    w = winoground_scores()
    put("t1.wino.group", w["group"], 78.8, label="Winoground group score")


# ------------------------------------------------------------------------------------------- Sec. 3-4 text, supp. table
def sec4() -> None:
    print("Sec. 3-4 text")
    Q, P = load("inquire_rerank_test")
    C = defaultdict(dict)       # SigLIP2's score for the query as written (search_score is the lowercased query's)
    for c in pq.read_table(data_root() / "inquire_rerank_test/candidates.parquet").to_pylist():
        C[c["query_id"]][c["image_id"]] = c["siglip2_score"]
    aps = [ap_at(sorted((p.image_id for p in P[q]), key=lambda i, q=q: -C[q][i]), relevant(Q[q]), 100) for q in P]
    put("s3.rerank.as_written", 100 * float(np.mean(aps)), 48.4, label="INQUIRE-Rerank, SigLIP2 with the query as written (AP)")
    put("s4.rerank.whole", rerank_ap("inquire_rerank_test", lambda ph: ranked_ids(ph, "whole")), 58.8, label="INQUIRE-Rerank, whole query alone")
    put("s4.rerank.whole_search", rerank_ap("inquire_rerank_test", lambda ph: ranked_ids(ph, "whole_search")), 62.5, label="  with the search score")
    for pool, v in (("vith_lc", 71.7), ("siglip2_lc", 72.1)):
        Q, P = load(f"inquire_full_{pool}_test")
        put(f"s4.recall500.{pool}", 100 * float(np.mean([len(relevant(Q[q]) & {p.image_id for p in P[q]}) / len(relevant(Q[q])) for q in P])), v,
            label=f"share of each query's relevant photos in {pool}'s top 500 (%)")
    Q, P = load("inquire_full_vith_lc_test")
    n50 = deep = 0
    for q in P:
        rel = relevant(Q[q])
        top = [p for p, _ in rank(P[q])][:50]
        n50 += sum(p.image_id in rel for p in top)
        deep += sum(p.image_id in rel and p.search_rank > 100 for p in top)
    put("s4.top50_from_below100", 100 * deep / n50, 17, 0, label="ViT-H, 500 checked: matches in our top 50 from below rank 100 (%)")
    yes = yes_rel = rej = rej_rel = 0
    for q in P:
        rel = relevant(Q[q])
        for p in P[q]:
            if p.search_rank <= 200:
                continue
            if (p.h or 0.0) >= 0.5:
                yes += 1
                yes_rel += p.image_id in rel
                if not all((x if x is not None else 0.0) >= 0.5 for x in p.s):
                    rej += 1
                    rej_rel += p.image_id in rel
    n = len(P)
    put("s4.below200.whole_yes", yes / n, 123, 0, label="below rank 200: whole query says yes, photos per query")
    put("s4.below200.whole_yes_match", 100 * yes_rel / yes, 17, 0, label="  of which matches (%)")
    put("s4.below200.statement_rejects", rej / n, 72, 0, label="  turned away by a statement, per query")
    put("s4.below200.statement_rejects_match", 100 * rej_rel / rej, 10, 0, label="  of which matches (%)")
    tok = [r["input_tokens"] for r in pq.read_table(data_root() / "inquire_rerank_test/answers.parquet", columns=["question_id", "input_tokens"]).to_pylist()
           if r["question_id"] == "holistic" and r["input_tokens"]]
    put("s3.cost.tokens_per_photo", float(np.mean(tok)), 1050, -1, label="INQUIRE-Rerank: input tokens per photo, seven questions")
    put("s3.cost.verifier_per_query", 100 * float(np.mean(tok)) * 0.042e-6 * 100, None, 3,
        label="  verifier cost per query, 100 photos at $0.042 per million (cents)")
    led = {r["role"]: r for r in pq.read_table(data_root() / "costs/ledgers.parquet").to_pylist() if r["source"] == "inquire_rerank_test"}
    usd = {k: (r["input_tokens"] * r["usd_per_million_input"] + r["output_tokens"] * r["usd_per_million_output"]) / 1e6 / r["queries"]
           for k, r in led.items()}
    put("s3.cost.verifier_usd", usd["verifier"], 0.0044, 4, label="INQUIRE-Rerank: verifier cost per query, from the run's ledger ($)")
    put("s3.cost.statements_usd", usd["statements"], 0.0027, 4, label="  writing the statements, per query ($)")
    put("s3.cost.total_usd", usd["verifier"] + usd["statements"], 0.0071, 4, label="  together ($)")
    from qr import bank as QB
    bt = QB.compare("bank_inquire_rerank_test")
    Qr, _ = load("inquire_rerank_test")
    b154 = QB.compare("bank_inquire_rerank_test", lambda r: meta(Qr[r["query_id"]]).get("supercategory") != "Species")
    bv, bd = QB.compare("bank_inquire_full_vith_lc_val"), QB.compare("bank_docci_s1")
    put("s4.bank.questions_per_photo", bt["questions_per_photo"], 15.7, label="larger question set: questions per photo (INQUIRE-Rerank)")
    put("s4.bank.rerank", 100 * bt["plain"]["families"], 67.5, label="  INQUIRE-Rerank AP with it (from 63.9)")
    put("s4.bank.rerank_gain", 100 * bt["plain.gain"][0], 3.6, label="  gain (Table 4: +3.6 on test)")
    put("s4.bank.rerank_gain_lo", 100 * bt["plain.gain"][1], 1.8, label="  95% CI, low")
    put("s4.bank.rerank_gain_hi", 100 * bt["plain.gain"][2], 5.5, label="  95% CI, high")
    put("s4.bank.brief_154", 100 * b154["brief"]["families"], 69.1, label="  with the LLM's brief, D&C's 154 queries (AP)")
    put("s4.bank.fullval_gain", 100 * bv["plain.gain"][0], 0.2, label="  Fullrank validation: gain (Table 4: +0.2)")
    put("s4.bank.docci_brief", 100 * bd["brief_minus_plain"][0], -2.7, label="  DOCCI: the brief's change (\"lowers DOCCI by 2.7\")")
    g = gpt4o_paired((100,))
    if g is None:
        print("  paired gain over GPT-4o: skipped, GPT-4o's answers not read (scripts/gpt4o_answers.py)")
    else:
        r = g[100]
        put("s4.gpt4o.n", r["n"], 160, 0, label="test queries with GPT-4o's released answers")
        put("s4.gpt4o.diff100", 100 * r["diff"], 2.6, label="ViT-H pool, 100 checked: paired gain over GPT-4o (mAP@50)")
        put("s4.gpt4o.lo100", 100 * r["lo"], 0.9, label="  95% CI, low")
        put("s4.gpt4o.hi100", 100 * r["hi"], 4.4, label="  95% CI, high")


def depth() -> None:
    print("Supplement: which answers make depth pay (INQUIRE-Fullrank test, mAP@50)")
    paper = {"vith_lc": {"whole": (43.7, 44.1, 42.5), "no_search": (45.0, 45.3, 43.9), "whole_search": (45.0, 46.8, 47.3),
                         "statements_search": (44.6, 46.3, 47.3), "whole_rating_search": (45.3, 47.4, 48.1), "paper": (46.2, 48.6, 49.7)},
             "siglip2_lc": {"whole": (44.3, 43.4, 42.0), "no_search": (44.7, 44.2, 43.1), "whole_search": (45.8, 46.5, 47.1),
                            "statements_search": (45.3, 46.4, 47.0), "whole_rating_search": (46.2, 47.2, 48.0), "paper": (46.7, 48.0, 49.0)}}
    for pool, rules in paper.items():
        src = f"inquire_full_{pool}_test"
        ap = {(r, k): fullrank_ap(src, k, r) for r in rules for k in KS}
        for r, vals in rules.items():
            for k, v in zip(KS, vals):
                put(f"depth.{pool}.{r}.{k}", 100 * ap[(r, k)].mean(), v, label=f"{pool}: {r}, {k} checked")
        ci = {("vith_lc", 100): (1.2, 0.6, 1.8), ("vith_lc", 500): (2.4, 1.4, 3.5), ("siglip2_lc", 100): (1.0, None, None),
              ("siglip2_lc", 500): (2.0, 1.1, 2.9)}       # the supplement table's caption
        for k in (100, 500):
            m, lo, hi = boot_ci(100 * (ap[("paper", k)] - ap[("whole_search", k)]))
            pm, plo, phi = ci[(pool, k)]
            put(f"depth.{pool}.lead_over_whole_search.{k}", m, pm, label=f"{pool}: full score minus whole query + search, {k} checked")
            put(f"depth.{pool}.lead_over_whole_search.{k}.lo", lo, plo, label="  95% CI low")
            put(f"depth.{pool}.lead_over_whole_search.{k}.hi", hi, phi, label="  95% CI high")


# ---------------------------------------------------------------------------------------------------------- Table 2
def docci_rows(source: str, answers_dir: str | None = None, rule: str = "paper", mode: str = "weighted"):
    """Per target: verifier right, verifier lead, search right, search margin."""
    Q, P = load(source, answers_dir=answers_dir)
    out = {}
    for q in P:
        w, a = RULES[rule]
        R = rank(P[q], a, w, mode)
        rel = relevant(Q[q])
        ph = P[q]
        out[q] = dict(ok=R[0][0].image_id in rel, lead=lead(R), emb_ok=ph[0].image_id in rel,
                      emb_lead=ph[0].search_score - ph[1].search_score)
    return out


def level_arrays(answers_dir=None, rule="paper", mode="weighted"):
    rows = {lv: docci_rows(f"docci_{lv}", answers_dir, rule, mode) for lv in LEVELS}
    T = sorted(set.intersection(*[set(r) for r in rows.values()]))
    arr = lambda key: {lv: np.array([rows[lv][t][key] for t in T]) for lv in LEVELS}  # noqa: E731
    return T, arr("ok"), arr("lead"), arr("emb_ok"), arr("emb_lead")


def table2() -> None:
    print("Table 2. Is the top result right?")
    T, ok, ld, eok, eld = level_arrays()
    v = sure_protocol(ld, ok, list(LEVELS))
    s = sure_protocol(eld, eok, list(LEVELS))
    paper = {"s1": (50.0, 38, 90.7, 16, 91.1), "s2": (76.2, 65, 96.4, 16, 98.2), "s3": (87.8, 77, 99.0, 19, 98.7),
             "s4": (92.8, 85, 99.6, 23, 98.8), "all": (98.4, 93, 99.9, 27, 100.0)}
    for lv in LEVELS:
        p = paper[lv]
        put(f"t2.{lv}.top", 100 * ok[lv].mean(), p[0], label=f"{lv}: top result right (%)")
        put(f"t2.{lv}.sure", 100 * v["sure"][lv], p[1], 0, label=f"{lv}: verifier's lead, sure for (%)")
        put(f"t2.{lv}.right", 100 * v["right"][lv], p[2], label=f"{lv}: verifier's lead, right (%)")
        put(f"t2.{lv}.margin_sure", 100 * s["sure"][lv], p[3], 0, label=f"{lv}: search margin, sure for (%)")
        put(f"t2.{lv}.margin_right", 100 * s["right"][lv], p[4], label=f"{lv}: search margin, right (%)")
    put("t2.pooled.top", 100 * np.mean([ok[lv].mean() for lv in LEVELS]), 81.0, label="pooled: top result right")
    put("t2.pooled.sure", 100 * v["sure_pooled"], 72, 0, label="pooled: verifier's lead, sure for")
    put("t2.pooled.right", 100 * v["right_pooled"], 98.0, label="pooled: verifier's lead, right")
    put("t2.pooled.margin_sure", 100 * s["sure_pooled"], 20, 0, label="pooled: search margin, sure for")
    put("t2.pooled.margin_right", 100 * s["right_pooled"], 97.8, label="pooled: search margin, right")
    tau = tau_for(np.concatenate([ld[lv] for lv in LEVELS]), np.concatenate([ok[lv] for lv in LEVELS]))
    etau = tau_for(np.concatenate([eld[lv] for lv in LEVELS]), np.concatenate([eok[lv] for lv in LEVELS]))
    put("t2.tau", tau, 0.114, 3, label="threshold from all 2,500 sentence-level queries")
    fin = np.array([t for t in v["taus"] if np.isfinite(t)])
    put("t2.tau_halves_mean", float(fin.mean()), 0.115, 3, label="  from half the targets: mean")
    put("t2.tau_halves_sd", float(fin.std()), None, 3, label="  sd")
    H = docci_rows("docci_heldout")
    hok = np.array([r["ok"] for r in H.values()]); hld = np.array([r["lead"] for r in H.values()])
    heok = np.array([r["emb_ok"] for r in H.values()]); held = np.array([r["emb_lead"] for r in H.values()])
    sure, esure = hld >= tau, held >= etau
    put("t2.heldout.n", float(len(H)), 500, 0, label="held out: targets")
    put("t2.heldout.top", 100 * hok.mean(), 98.6, label="held out: top result right (%)")
    put("t2.heldout.sure", 100 * sure.mean(), 95, 0, label="held out: verifier's lead, sure for")
    put("t2.heldout.right", 100 * hok[sure].mean(), 99.8, label="held out: verifier's lead, right")
    put("t2.heldout.right_n", float(hok[sure].sum()), 475, 0, label="  right")
    put("t2.heldout.sure_n", float(sure.sum()), 476, 0, label="  of sure")
    put("t2.heldout.margin_sure", 100 * esure.mean(), 35, 0, label="held out: search margin, sure for")
    put("t2.heldout.margin_right", 100 * heok[esure].mean(), 96.6, label="held out: search margin, right")


# ---------------------------------------------------------------------------------------------------- Sec. 5 text
def sec5() -> None:
    print("Sec. 5 text")
    T, ok, ld, eok, eld = level_arrays()
    put("s5.auroc.s1", auroc(ld["s1"], ok["s1"]), 0.87, 2, label="lead separates right from wrong, one sentence (AUROC)")
    put("s5.auroc.all", auroc(ld["all"], ok["all"]), 0.99, 2, label="  all sentences")
    # the lead with fewer questions: lead_ablation.py's protocol (rows pooled, halves of the targets from one generator)
    for rule, paper in (("whole", 65), ("whole_search", 69), ("paper", 72)):
        _, o, l, _, _ = level_arrays(rule=rule)
        okp = np.concatenate([o[lv] for lv in LEVELS]); v = np.concatenate([l[lv] for lv in LEVELS])
        tg = np.concatenate([np.arange(len(T))] * len(LEVELS))
        rng = np.random.default_rng(0); sf = []
        for _ in range(40):
            perm = rng.permutation(len(T)); h = len(T) // 2
            for a_, b_ in ((perm[:h], perm[h:]), (perm[h:], perm[:h])):
                ia = np.isin(tg, a_)
                t = tau_for(v[ia], okp[ia]); sf.append((v[~ia] >= t).mean())
        put(f"s5.lead_ablation.{rule}.sure", 100 * float(np.mean(sf)), paper, 0, label=f"lead from {rule}: sure for (%)")
    for lv, share, right in (("s1", 24, 14), ("s2", 4, None)):
        Q, P = load(f"docci_{lv}")
        vague = [q for q in P if sum(passes_all(p) for p in P[q]) >= 5]
        put(f"s5.vague.{lv}.share", 100 * len(vague) / len(P), share, 0, label=f"{lv}: queries with five or more photos passing every question (%)")
        if right is not None:
            put(f"s5.vague.{lv}.top_right", 100 * float(np.mean([ranked_ids(P[q])[0] in relevant(Q[q]) for q in vague])), right, 0,
                label="  top photo right for those (%)")
    w = winoground_scores()
    put("s5.wino.auroc_min_margin", w["auroc_min_margin"], 0.93, 2, label="Winoground: smallest of four margins, right vs wrong groups (AUROC)")


# ---------------------------------------------------------------------------------------------------------- Table 3
def table3() -> None:
    """Table 3 (R@1 and mean sentences) and the loop's breakdown in Sec. 6. Sentences used: min(level, sentences in the
    description) at a level, all of them at the last; the fixed 4-sentence budget is printed as its nominal 4.00."""
    print("Table 3. Retrieval accuracy and description use on DOCCI")
    T, ok, ld, eok, eld = level_arrays()
    tg = {r["query_id"]: r["n_sentences"] for r in pq.read_table(data_root() / "docci/targets.parquet").to_pylist()}
    used = {lv: np.array([min(int(lv[1]), tg[t]) if lv != "all" else tg[t] for t in T], float) for lv in LEVELS}
    from qr.metrics import ask_until_sure
    a = ask_until_sure(ld, ok, used, list(LEVELS))
    e = ask_until_sure(eld, eok, used, list(LEVELS))
    put("t3.every2.exact", 100 * ok["s2"].mean(), 76.2, label="fixed: 2 sentences, R@1 (%)")
    put("t3.every2.sentences", used["s2"].mean(), 2.00, 2, label="  mean sentences")
    put("t3.every4.exact", 100 * ok["s4"].mean(), 92.8, label="fixed: 4 sentences, R@1 (%)")
    put("t3.every4.sentences", used["s4"].mean(), None, 2, label="  mean sentences actually used (the table prints the budget, 4.00)")
    put("t3.every4.short_descriptions", float((used["all"] < 4).sum()), None, 0, label="  descriptions with fewer than four sentences")
    put("t3.everyall.exact", 100 * ok["all"].mean(), 98.4, label="fixed: full description, R@1 (%)")
    put("t3.everyall.sentences", used["all"].mean(), 7.29, 2, label="  mean sentences")
    put("t3.ask_margin.exact", 100 * e["exact"], 75.5, label="adaptive: margin threshold, SigLIP2, R@1 (%)")
    put("t3.ask_margin.sentences", e["sentences"], 5.76, 2, label="  mean sentences")
    put("t3.ask_lead.exact", 100 * a["exact"], 93.0, label="adaptive: margin threshold, Jev + SigLIP2, R@1 (%)")
    put("t3.ask_lead.sentences", a["sentences"], 2.55, 2, label="  mean sentences")
    # the loop's breakdown (Sec. 6; ask_loop_stats.py): why it stopped, and how often it was right
    n = len(T)
    crossed = right_c = exhausted = right_e = runs = 0
    n_photos = []
    for tune, test in halves(n):
        tau = tau_for(np.concatenate([ld[lv][tune] for lv in LEVELS]), np.concatenate([ok[lv][tune] for lv in LEVELS]))
        runs += 1
        for i in test:
            stop = next((lv for lv in LEVELS if ld[lv][i] >= tau), None)
            if stop is None:
                exhausted += 1; right_e += ok["all"][i]
            else:
                crossed += 1; right_c += ok[stop][i]
            n_photos.append(100 * (LEVELS.index(stop) + 1 if stop else len(LEVELS)))
    put("s6.loop.crossed", 100 * crossed / (crossed + exhausted), 95.8, label="Sec. 6: interactions that end because the margin crosses (%)")
    put("s6.loop.crossed_right", 100 * right_c / crossed, 93.6, label="  those returns right (%)")
    put("s6.loop.exhausted", 100 * exhausted / (crossed + exhausted), 4.2, label="  interactions that exhaust the description (%)")
    put("s6.loop.exhausted_right", 100 * right_e / exhausted, 79.3, label="  those return the target (%)")
    put("t3.ask_lead.photos", float(np.mean(n_photos)), None, 0, label="  photos checked per interaction (100 per round; not in the paper)")
    expand_before_asking(T, ok, ld, tg)
    stop_probability()


def expand_before_asking(T, ok, ld, tg) -> None:
    """Table 3's last row (docci_deep_report.py): at each level answer when the first 100 are sure; if not, add the 100
    photos the answers chose, rank the pool again and answer if sure; else ask for the next sentence. After four
    sentences, answer with the full description (100 photos). The threshold is Table 2's, from the other half."""
    lv4 = LEVELS[:-1]
    leadS, okS = {}, {}
    for lv in lv4:
        src = f"docci_deep200_{lv}"
        Q, P = load(src)
        st = defaultdict(set)
        for r in pq.read_table(data_root() / src / "steered.parquet").to_pylist():
            st[r["query_id"]].add(r["image_id"])
        L, O = [], []
        for t in T:
            first = [p for p in P[t] if p.search_rank <= 100]
            chosen = [p for p in P[t] if p.image_id in st[t] and p.search_rank > 100]
            R = rank(first + chosen)
            L.append(lead(R)); O.append(R[0][0].image_id in relevant(Q[t]))
        leadS[lv], okS[lv] = np.array(L), np.array(O)
    used = {lv: np.array([min(int(lv[1]), tg[t]) for t in T], float) for lv in lv4}
    used_all = np.array([tg[t] for t in T], float)
    acc, nu = [], []
    for tune, test in halves(len(T)):
        tau = tau_for(np.concatenate([ld[lv][tune] for lv in LEVELS]), np.concatenate([ok[lv][tune] for lv in LEVELS]))
        done = np.zeros(len(test), bool); right = np.zeros(len(test), bool); u = np.zeros(len(test))
        for lv in lv4:
            for L_, O_ in ((ld[lv], ok[lv]), (leadS[lv], okS[lv])):
                s = (L_[test] >= tau) & ~done
                right[s] = O_[test][s]; u[s] = used[lv][test][s]; done |= s
        rest = ~done
        right[rest] = ok["all"][test][rest]; u[rest] = used_all[test][rest]
        acc.append(right.mean()); nu.append(u.mean())
    put("t3.expand.exact", 100 * float(np.mean(acc)), 93.0, label="adaptive: expand before asking, R@1 (%)")
    put("t3.expand.sentences", float(np.mean(nu)), 2.42, 2, label="  mean sentences")


def stop_probability() -> None:
    """Eq. 3 on INQUIRE-Fullrank (ViT-H pool): a calibrated match probability per photo, fitted on validation, then
    p_next from the last 100 photos checked, fitted on the validation queries and used unchanged on test."""
    def geo(S, w):
        S = np.clip(np.nan_to_num(S, nan=.5), .02, 1)
        return np.exp((np.log(S) * w).sum(1) / w.sum())

    def lg(p):
        p = np.clip(p, .02, .98)
        return np.log(p / (1 - p))

    def logreg(X, y, l2=1e-3):
        X = np.asarray(X, float); y = np.asarray(y, float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.c_[(X - mu) / sd, np.ones(len(X))]; w = np.zeros(A.shape[1])
        for _ in range(80):
            p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
            g = A.T @ (p - y) / len(y) + l2 * w
            H = (A * (p * (1 - p))[:, None]).T @ A / len(y) + l2 * np.eye(A.shape[1])
            st = np.linalg.solve(H, g); w -= st
            if np.abs(st).max() < 1e-8:
                break
        return lambda Z: 1 / (1 + np.exp(-np.clip(np.c_[(np.asarray(Z, float) - mu) / sd, np.ones(len(Z))] @ w, -30, 30)))

    def queries(split):
        src = f"inquire_full_vith_lc_{split}"
        Q, P = load(src)
        fs = defaultdict(list)
        for r in pq.read_table(data_root() / src / "firststage.parquet").to_pylist():
            fs[r["query_id"]].append(r)
        ans = defaultdict(dict)
        for r in pq.read_table(data_root() / src / "answers.parquet", columns=["query_id", "image_id", "question_id", "p_levels"]).to_pylist():
            if r["question_id"] == "relevance" and r["p_levels"]:
                ans[r["query_id"]][r["image_id"]] = int(np.argmax(r["p_levels"]) == 4)
        out = []
        for q in sorted(P):
            ids = [r["image_id"] for r in sorted(fs[q], key=lambda r: r["rank"])][:1000]
            by = {p.image_id: p for p in P[q]}
            ph = [by[i] for i in ids[:500]]
            h = np.array([.5 if p.h is None else p.h for p in ph])
            r = np.array([.5 if p.r is None else p.r for p in ph])
            S = np.array([[np.nan if x is None else x for x in p.s] for p in ph], float)
            out.append(dict(y=np.array([i in relevant(Q[q]) for i in ids]), h=h, r=r, s=geo(S, np.array(ph[0].w)),
                            L5=np.array([ans[q].get(p.image_id, 0) for p in ph], bool), nrel=len(relevant(Q[q])), n=len(ids)))
        return out

    V, T = queries("val"), queries("test")
    cal = logreg(np.concatenate([np.c_[lg(q["h"]), lg(q["s"]), q["r"]] for q in V]), np.concatenate([q["y"][:500] for q in V]))
    for q in V + T:
        q["p"] = cal(np.c_[lg(q["h"]), lg(q["s"]), q["r"]])
    # Sec. 5: calibration of the match probability over the 100,000 test photos
    p = np.concatenate([q["p"] for q in T]); hh = np.concatenate([q["h"] for q in T]); yy = np.concatenate([q["y"][:500] for q in T])
    ece = lambda pr: float(sum(abs(pr[m].mean() - yy[m].mean()) * m.mean() for m in (  # noqa: E731
        (pr >= lo) & ((pr < lo + .1) if lo < .9 else (pr <= 1)) for lo in np.arange(0, 1, .1)) if m.any()))
    put("s5.ece.whole", ece(hh), 0.32, 2, label="INQUIRE test, 100,000 photos: ECE of P(yes) to the whole query")
    put("s5.ece.calibrated", ece(p), 0.02, 2, label="  ECE of the calibrated match probability")
    put("s5.calibrated_above_0.9.n", float((p > .9).sum()), 198, 0, label="  photos above 0.9")
    put("s5.calibrated_above_0.9.right", 100 * float(yy[p > .9].mean()), 95, 0, label="  relevant (%)")

    def rows(Qs):
        R = []
        for q in Qs:
            for D in (100, 200, 300, 400, 500):
                if q["n"] < D + 100:
                    continue
                a, b = D - 100, D
                R.append(([q["p"][a:b].sum(), q["L5"][a:b].mean(), q["h"][a:b].mean(), D / 100.0], bool(q["y"][D:D + 100].any())))
        return np.array([x for x, _ in R]), np.array([y for _, y in R])
    Xv, yv = rows(V); Xt, yt = rows(T)
    P_ = logreg(Xv, yv, l2=1e-2)(Xt)
    put("s6.next_batch.points", float(len(yt)), 1000, 0, label="Sec. 6 next-batch predictor: (query, depth) points on test")
    put("s6.next_batch.auroc", auroc(P_, yt), 0.80, 2, label="  AUROC")
    for lo, hi, v in ((.2, .4, 26), (.95, 1.01, 96)):
        m = (P_ >= lo) & (P_ < hi)
        put(f"s6.next_batch.bin_{int(100 * lo)}_{int(min(100, 100 * hi))}", 100 * float(yt[m].mean()), v, 0, label=f"  says {100 * lo:.0f}-{min(100, 100 * hi):.0f}%: true rate")
    # Table 4, counts: the summed calibrated probabilities over the 500 checked, against the true number
    tot = np.array([q["p"][:500].sum() for q in T]); n500 = np.array([q["y"][:500].sum() for q in T])
    put("t4.counts.mae", float(np.abs(tot - n500).mean()), 39, 0, label="Table 4: count of matches in the 500, off per query by")
    put("t4.counts.true_mean", float(n500.mean()), 64.7, label="  on a mean of")


# ---------------------------------------------------------------------------------------------------- Sec. 6 text
def sec6() -> None:
    """Looking further on DOCCI: the search's next 100 (ranks 101-200) against the 100 photos the answers chose
    (steered.parquet), each added to the first 100 and ranked together."""
    print("Sec. 6 text: candidate expansion")
    for lv in ("s1", "s2", "s3", "s4"):
        src = f"docci_deep200_{lv}"
        Q, P = load(src)
        st = defaultdict(set)
        for r in pq.read_table(data_root() / src / "steered.parquet").to_pylist():
            st[r["query_id"]].add(r["image_id"])
        found_s = found_a = recovered_first = 0
        ex_s, ex_a, ex_0 = [], [], []
        for q in sorted(P):
            rel = relevant(Q[q])
            first = [p for p in P[q] if p.search_rank <= 100]
            nxt = [p for p in P[q] if 100 < p.search_rank <= 200]
            chosen = [p for p in P[q] if p.image_id in st[q] and p.search_rank > 100]
            missing = not any(p.image_id in rel for p in first)
            back = missing and any(p.image_id in rel for p in chosen)
            found_s += missing and any(p.image_id in rel for p in nxt)
            found_a += back
            top_a = rank(first + chosen)[0][0].image_id in rel
            recovered_first += back and top_a
            ex_0.append(rank(first)[0][0].image_id in rel)
            ex_s.append(rank(first + nxt)[0][0].image_id in rel)
            ex_a.append(top_a)
        ex_0, ex_s, ex_a = (np.array(x, float) for x in (ex_0, ex_s, ex_a))
        put(f"s6.{lv}.recovered_first", 100 * recovered_first / max(1, found_a), {"s2": 50, "s4": 92}.get(lv), 0,
            label=f"{lv}: targets the answers' 100 recover that then rank first (%)")
        if lv in ("s3", "s4"):
            continue
        if lv == "s1":
            put("s6.s1.found_by_answers", float(found_a), 40, 0, label="one sentence: missing targets the answers' next 100 bring in")
            put("s6.s1.found_by_search", float(found_s), 27, 0, label="  the search's next 100")
        put(f"s6.{lv}.exact_first100", 100 * ex_0.mean(), {"s1": 50.0, "s2": 76.2}[lv], label=f"{lv}: exact, first 100 photos")
        put(f"s6.{lv}.exact_answers", 100 * ex_a.mean(), {"s1": 51.2, "s2": None}[lv], label=f"{lv}: exact, + the answers' next 100")
        put(f"s6.{lv}.exact_search", 100 * ex_s.mean(), None, label=f"{lv}: exact, + the search's next 100")
        if lv == "s2":   # docci_deep_report.py: 2,000 resamples
            m, lo, hi = boot_ci(100 * (ex_a - ex_0), n_boot=2000)
            put("s6.s2.gain_answers", m, 3.2, label="two sentences: the answers' next 100 add (points)")
            m, lo, hi = boot_ci(100 * (ex_a - ex_s), n_boot=2000)
            put("s6.s2.answers_minus_search", m, 2.0, label="  more than the search's next 100")
            put("s6.s2.answers_minus_search.lo", lo, 0.8, label="  95% CI low")
            put("s6.s2.answers_minus_search.hi", hi, 3.2, label="  95% CI high")
    _, ok, _, _, _ = level_arrays()
    for a_, b_, v in (("s1", "s2", 26.2), ("s2", "s3", 11.6), ("s3", "s4", 5.0)):
        put(f"s6.sentence_gain.{b_}", 100 * (ok[b_].mean() - ok[a_].mean()), v, label=f"one more sentence, {a_} to {b_}: R@1 gain (points)")


# ---------------------------------------------------------------------------------------- Sec. 7 text and Table 4
KINDS = ("relevant", "same species", "same genus", "same family", "other")


def _taxon(category: str) -> tuple[str, str, str]:
    """(species, genus, family) from iNat24's folder name: category id, kingdom, phylum, class, order, family, genus,
    species."""
    p = category.split("_")
    return ("_".join(p[1:]), "_".join(p[1:-1]), "_".join(p[1:-2])) if len(p) >= 8 else (category, category, category)


def species_errors(split: str = "test") -> dict:
    """Sec. 7 and Table 4 on INQUIRE-Fullrank, CLIP ViT-H's top 500 (limits_final.py): each photo's kind against the
    query's relevant photos among the 500 (relevant; same species but not relevant; same genus; same family; other),
    the share of each kind in the top 50, how often each kind passes every question, and mAP@50 when photos of another
    species, or of the right species in the wrong state or view, are moved below the others with the labels."""
    cat = dict(zip(*pq.read_table(data_root() / "inquire_taxa/taxa.parquet", columns=["image_id", "category"]).to_pydict().values()))
    Q, P = load(f"inquire_full_vith_lc_{split}")
    top, passn, passd = np.zeros(5), np.zeros(5), np.zeros(5)
    ap7, known, state = [], [], []
    for q in sorted(P):
        rel = relevant(Q[q])
        R = rank(P[q])
        ids = [p.image_id for p, _ in R]
        fin = np.array([s for _, s in R])
        y = np.array([i in rel for i in ids])
        tx = [_taxon(cat[i]) for i in ids]
        sp, ge, fa = ({t[k] for t, yy in zip(tx, y) if yy} for k in range(3))
        if not sp:                         # no relevant photo among the 500: every ranking scores 0
            ap7.append(0.0); known.append(0.0); state.append(0.0)
            continue
        kind = np.array([0 if yy else 1 if t[0] in sp else 2 if t[1] in ge else 3 if t[2] in fa else 4 for t, yy in zip(tx, y)])
        order = lambda score: [ids[i] for i in np.argsort(-score, kind="stable")]  # noqa: E731
        ap7.append(ap_at(ids, rel, 50))
        known.append(ap_at(order(np.where(kind <= 1, fin, -1.0)), rel, 50))
        state.append(ap_at(order(np.where(kind != 1, fin, -1.0)), rel, 50))
        for k in kind[:50]:
            top[k] += 1
        ok = np.array([passes_all(p) for p, _ in R])
        for k in range(5):
            passn[k] += ok[kind == k].sum(); passd[k] += (kind == k).sum()
    return dict(seven=float(np.mean(ap7)), species_known=float(np.mean(known)), state_view=float(np.mean(state)),
                top50={KINDS[k]: float(top[k] / top.sum()) for k in range(5)}, other_species=float(top[2:].sum() / top.sum()),
                passes={KINDS[k]: float(passn[k] / max(1, passd[k])) for k in range(5)}, queries=int(top.sum() // 50))


def sec7() -> None:
    """The rows of Table 4 and the Sec. 7 numbers that the answers alone determine (INQUIRE-Fullrank test, ViT-H's
    top 500; DOCCI)."""
    print("Sec. 7 and Table 4")
    for lv, v in (("s1", 0.59), ("s2", None), ("s3", None), ("s4", 0.99)):
        Q, P = load(f"docci_{lv}")
        top, present = [], []
        for q in sorted(P):
            rel = relevant(Q[q])
            R = rank(P[q])
            if R[0][0].image_id in rel:
                continue                         # among queries whose top photo is wrong:
            top.append(R[0][1])                  # the top score tells a target missing from the 100
            present.append(any(p.image_id in rel for p in P[q]))   # from one in the 100 but not first
        put(f"t4.missing_target.{lv}", auroc(top, present), v, 2, label=f"detect a missing target, {lv}: AUROC of the top score")
    Q, P = load("inquire_full_vith_lc_test")
    perfect, best = [], []
    for q in sorted(P):
        rel = relevant(Q[q])
        n_in = sum(p.image_id in rel for p in P[q])
        perfect.append(min(n_in, 50) / min(len(rel), 50) if rel else 0.0)
        if not n_in:
            best.append(0.0)
            continue
        base = [p for p, _ in rank(P[q])]            # ties keep the full score's order (limits_final.py)
        sc = np.array([p.search_score for p in base]); cm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-12)
        cols = [np.array([p.h or 0.0 for p in base])] + [np.array([(p.s[j] if p.s[j] is not None else 0.0) for p in base])
                                                         for j in range(len(base[0].s))] + [np.array([p.r or 0.0 for p in base])]
        best.append(max(ap_at([base[i].image_id for i in np.argsort(-(0.8 * v_ + 0.2 * cm), kind="stable")], rel, 50) for v_ in cols))
    put("t4.perfect_verifier", 100 * float(np.mean(perfect)), 87.0, label="perfect verification of ViT-H's top 500 (mAP@50)")
    put("t4.best_question", 100 * float(np.mean(best)), 54.1, label="best one of the seven questions per query, chosen with labels (mAP@50)")
    e = species_errors("test")
    put("t4.species.seven", 100 * e["seven"], 49.7, label="species errors: the ranking itself (mAP@50)")
    put("t4.species.known", 100 * e["species_known"], 70.7, label="  with the species known (labels move other species down)")
    put("t4.species.state_view", 100 * e["state_view"], 56.5, label="  with state and view known instead")
    put("t4.species.pass_genus", 100 * e["passes"]["same genus"], 19, 0, label="  same genus: passes every question (%)")
    put("t4.species.pass_family", 100 * e["passes"]["same family"], 23, 0, label="  same family: passes every question (%)")
    put("limits.species.other_species_top50", 100 * e["other_species"], 38.8, label="  top 50: another species (%) (Sec. 7: more than a third)")
    put("limits.species.top50_relevant", 100 * e["top50"]["relevant"], 44.0, label="  top 50: relevant (%)")
    put("limits.species.top50_same_species", 100 * e["top50"]["same species"], 17.2, label="  top 50: right species, not relevant (%)")
    put("limits.species.pass_relevant", 100 * e["passes"]["relevant"], 63, 0, label="  relevant photos passing every question (%)")
    put("limits.species.pass_same_species", 100 * e["passes"]["same species"], 11, 0, label="  same species, not relevant, passing (%)")
    from qr import uncertainty as UN
    u, uv = UN.summary("test"), UN.summary("val")
    put("t4.unclear.chosen_low", u["chosen"][0], 0, 0, label="report uncertainty: 'too unclear' the likeliest answer, lowest group (%)")
    put("t4.unclear.chosen_high", u["chosen"][1], 3, 0, label="  highest group (%) (Table 4: 0-3%)")
    put("limits.unclear.wrongly_rejected.plain", u["wrongly_rejected"][0], 6.9, label="  can't-judge evidence, wrongly rejected matches, plainly (%)")
    put("limits.unclear.wrongly_rejected.brief", u["wrongly_rejected"][1], 7.2, label="  with the brief (%)")
    put("limits.unclear.correctly_rejected.plain", u["correctly_rejected"][0], 15.6, label="  correctly rejected photos, plainly (%)")
    put("limits.unclear.correctly_rejected.brief", u["correctly_rejected"][1], 18.3, label="  with the brief (%)")
    for k, v, nd in (("chosen_low", uv["chosen"][0], 0), ("chosen_high", uv["chosen"][1], 0), ("wrongly_rejected.plain", uv["wrongly_rejected"][0], 1),
                     ("wrongly_rejected.brief", uv["wrongly_rejected"][1], 1), ("correctly_rejected.plain", uv["correctly_rejected"][0], 1),
                     ("correctly_rejected.brief", uv["correctly_rejected"][1], 1)):
        put(f"limits.unclear.val.{k}", v, None, nd, label=f"  validation: {k.replace('_', ' ').replace('.', ', ')} (%)")
    from qr import feedback as FB
    rw, lt = FB.rewording(), FB.llm_text()
    put("t4.rewording.union", 100 * rw["test"][0], 49.9, label="rewording: relevant photos in the union of the searches' top 20 (%)")
    put("t4.rewording.own", 100 * rw["test"][1], 55.4, label="  in the query's own ranking at the same number of photos (%)")
    put("t4.llm_text.test", 100 * lt["test"][2], 3.9, label="LLM text with the query: more relevant photos in the next 100, test (points)")
    put("t4.llm_text.val", 100 * lt["val"][2], 0.1, label="  validation (points)")
    st = FB.steer(100, (100,))
    t, v = st["test"][100], st["val"][100]
    put("t4.steer.gain", 100 * t["gain"][0], 1.3, label="steered by the answers: more relevant photos in the next 100 (points)")
    put("t4.steer.labels_gain", 100 * t["labels_gain"], 6.7, label="  steered with the labels instead (points)")
    put("limits.steer.text", 100 * t["text"], 14.9, label="  SigLIP2's search for the query, next 100 (% of relevant)")
    put("limits.steer.steered", 100 * t["steered"], 16.2, label="  steered by the answers (%)")
    put("limits.steer.gain_low", 100 * t["gain"][1], 0.2, label="  difference, 95% interval: low")
    put("limits.steer.gain_high", 100 * t["gain"][2], 2.3, label="  high")
    put("limits.steer.labels", 100 * t["labels"], 21.6, label="  steered with the labels (%)")
    put("limits.steer.val.text", 100 * v["text"], 15.1, label="  validation: SigLIP2's search (%)")
    put("limits.steer.val.steered", 100 * v["steered"], 17.9, label="  validation: steered (%)")
    put("limits.steer.val.gain", 100 * v["gain"][0], 2.9, label="  validation: difference (points)")
    put("limits.steer.val.gain_low", 100 * v["gain"][1], 1.0, label="  low")
    put("limits.steer.val.gain_high", 100 * v["gain"][2], 4.7, label="  high")
    put("limits.steer.val.labels_gain", 100 * v["labels_gain"], 5.6, label="  validation: with the labels (points)")


# ---------------------------------------------------------------------------------------------------------- figures
def figures() -> None:
    print("Figures 2 and 3")
    T, ok, ld, _, _ = level_arrays(rule="whole_search")
    put("fig2.tau_whole_search", tau_for(np.concatenate([ld[lv] for lv in LEVELS]), np.concatenate([ok[lv] for lv in LEVELS])), 0.155, 3,
        label="Fig. 2: threshold for the whole query + search (from all 2,500)")
    Q, P = load("docci_s2")
    q = "docci1923"
    rel = relevant(Q[q])
    full, ws = rank(P[q]), rank(P[q], *RULES["whole_search"][::-1])
    put("fig2.lead_full", lead(full), 0.14, 2, label="  docci1923, two sentences: lead with all seven questions")
    put("fig2.lead_whole_search", lead(ws), 0.06, 2, label="  lead with the whole query + search")
    tgt = next(p for p in P[q] if p.image_id in rel)
    f = dict((p.image_id, s) for p, s in full)
    put("fig2.score_target", f[tgt.image_id], 0.80, 2, label="  verifier score, target")
    put("fig2.score_search_first", f[P[q][0].image_id], 0.65, 2, label="  verifier score, search's first pick")
    put("fig2.target_search_rank", float(tgt.search_rank), 7, 0, label="  target's search rank")
    Q, P = load("inquire_full_vith_lc_test")
    rel = relevant(Q["test_135"])
    top = [p for p, _ in rank(P["test_135"])][:50]
    put("fig3.top50_matches", float(sum(p.image_id in rel for p in top)), 49, 0, label="Fig. 3, test_135: matches in the verifier's top 50")
    put("fig3.top50_below100", float(sum(p.image_id in rel and p.search_rank > 100 for p in top)), 32, 0, label="  from below search rank 100")


def examples() -> None:
    """The numbers printed in the supplement's example figures (S4: one description through the interaction; S5: six
    look-alikes), against the supplement's text."""
    print("Supplement: example figures")
    tg = {r["query_id"]: r["n_sentences"] for r in pq.read_table(data_root() / "docci/targets.parquet").to_pylist()}
    q = "docci4982"
    Q, P = load("docci_all")
    R = rank(P[q])
    tgt = next(p for p, _ in R if p.image_id in relevant(Q[q]))
    first = sorted(P[q], key=lambda p: (-p.search_score, p.search_rank))[0]
    f = {p.image_id: s for p, s in R}
    put("examples.anole.sentences", float(tg[q]), 11, 0, label="Fig. S4, docci4982: sentences in the description")
    put("examples.anole.sim_gap", first.search_score - tgt.search_score, 0.0001, 4, label="  search puts a look-alike first, by (similarity)")
    put("examples.anole.f_target", f[tgt.image_id], 0.88, 2, label="  Jev + SigLIP2, target")
    put("examples.anole.f_lookalike", f[first.image_id], 0.50, 2, label="  Jev + SigLIP2, the look-alike")
    for lv, printed in (("s1", None), ("s2", None), ("s3", None), ("s4", 0.15)):
        Ql, Pl = load(f"docci_{lv}")
        Rl = rank(Pl[q])
        right = Rl[0][0].image_id in relevant(Ql[q])
        put(f"examples.anole.lead_{lv}", lead(Rl), printed, 3, label=f"  lead with {lv[1]} sentence(s) ({'right' if right else 'wrong'} first; threshold 0.114)")
    gaps = []
    for lv, t in (("all", "docci3262"), ("all", "docci72"), ("all", "docci3048"), ("s2", "docci4470"), ("s3", "docci3430")):
        Ql, Pl = load(f"docci_{lv}")
        o = sorted(Pl[t], key=lambda p: (-p.search_score, p.search_rank))
        Rl = rank(Pl[t])
        assert o[1].image_id in relevant(Ql[t]) and Rl[0][0].image_id in relevant(Ql[t])
        gaps.append(o[0].search_score - o[1].search_score)
    put("examples.lookalikes.max_gap", max(gaps), 0.0004, 4, label="Fig. S5: search's first pick within this of the target (five panels)")
    Ql, Pl = load("docci_all")
    put("examples.lookalikes.last_lead", lead(rank(Pl["docci753"])), 0.01, 2, label="  last panel: the verifier's lead for a wrong photo")


# ------------------------------------------------------------------------------------------------- Clef, fusion
def clef() -> None:
    print("Supplement: the open verifier Clef-Flash, same questions, 100 photos")
    paper = {"docci_s1": 49.6, "docci_all": 97.0}
    for lv in LEVELS:
        put(f"clef.docci.{lv}", docci_top1(f"docci_{lv}", "paper", "clef_flash"), paper.get(f"docci_{lv}"), label=f"DOCCI {lv}: exact first (%)")
    Q, P = load("inquire_rerank_test", answers_dir="clef_flash")
    put("clef.rerank", 100 * float(np.mean([ap_at([p.image_id for p, _ in rank(P[q])], relevant(Q[q]), 100) for q in P])), 63.4,
        label="INQUIRE-Rerank (AP)")
    for pool in ("vith_lc", "siglip2_lc"):
        Q, P = load(f"inquire_full_{pool}_test", answers_dir="clef_flash")
        put(f"clef.full.{pool}.100", 100 * float(np.mean([ap_at([p.image_id for p, _ in rank([x for x in P[q] if x.search_rank <= 100])],
                                                               relevant(Q[q]), 50) for q in sorted(P)])), None,
            label=f"INQUIRE-Fullrank {pool}, 100 checked (mAP@50)")
    T, ok, ld, _, _ = level_arrays(answers_dir="clef_flash")
    v = sure_protocol(ld, ok, list(LEVELS))
    put("clef.lead.sure", 100 * v["sure_pooled"], 68, 0, label="lead, threshold on its own scale: sure for (%)")
    put("clef.lead.right", 100 * v["right_pooled"], 98.0, label="  right (%)")
    fin = np.array([t for t in v["taus"] if np.isfinite(t)])
    put("clef.lead.tau_mean", float(fin.mean()), None, 3, label="  threshold from half the targets: mean")


def fusion() -> None:
    print("Supplement: fusion without weights (each signal standardised over the pool, summed)")
    put("fusion.rerank", rerank_ap("inquire_rerank_test", lambda ph: [p.image_id for p, _ in rank(ph, mode="combsum_z")]), None,
        label="INQUIRE-Rerank (AP), against 63.9")
    for lv in LEVELS:
        Q, P = load(f"docci_{lv}")
        put(f"fusion.docci.{lv}", 100 * float(np.mean([rank(P[q], mode="combsum_z")[0][0].image_id in relevant(Q[q]) for q in P])), None,
            label=f"DOCCI {lv}: exact first (%)")
    _, ok, _, _, _ = level_arrays(mode="combsum_z")
    put("fusion.docci.pooled", 100 * float(np.mean([ok[lv].mean() for lv in LEVELS])), 80.7, label="DOCCI pooled exact first, against 81.0")


GROUPS = {"table1": table1, "sec4": sec4, "depth": depth, "table2": table2, "sec5": sec5, "table3": table3, "sec6": sec6,
          "sec7": sec7, "figures": figures, "examples": examples, "clef": clef, "fusion": fusion}

if __name__ == "__main__":
    want = sys.argv[1:] or list(GROUPS)
    for g in want:
        GROUPS[g]()
    f = ROOT / "results/numbers.json"
    f.parent.mkdir(exist_ok=True)
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **OUT}, indent=1, sort_keys=True))
    main = [r for r in ROWS if r[5] is not None and not r[0].startswith(SUPP)]
    supp = [r for r in ROWS if r[5] is not None and r[0].startswith(SUPP)]
    bad = [r for r in main + supp if not r[5]]
    new = sum(r[3] is None for r in ROWS)
    print(f"\n{sum(r[5] for r in main)} of {len(main)} numbers printed in the paper match"
          + (f"; {sum(r[5] for r in supp)} of {len(supp)} values of the draft supplement tables match" if supp else "")
          + (f"; {new} more values computed for the supplement" if new else "")
          + (". Differ: " + ", ".join(r[0] for r in bad) if bad else ""))
