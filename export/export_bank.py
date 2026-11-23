#!/usr/bin/env python
"""The larger question set ("the bank", Sec. 4 and Table 4): its questions and the verifier's answers, added to the
data folder as five sources, with what the comparisons need from the run each set was asked on.

  bank_inquire_rerank_test       INQUIRE-Rerank test, 160 queries x 100 photos, asked plainly and with the LLM's brief
  bank_inquire_full_vith_lc_val  INQUIRE-Fullrank validation, 50 queries x CLIP ViT-H's top 500, asked plainly
  bank_docci_s1                  DOCCI, the first sentence of 150 descriptions x 100 photos, plainly and with the brief
  bank_inquire_wide_test         INQUIRE-Fullrank test, 200 queries x every photo checked for the query in four earlier
  bank_inquire_wide_val          runs (about 900: the query's own top 500 with CLIP ViT-H and with SigLIP2, and the
                                 pools of two runs that searched with 45 LLM-written texts per query), plainly and
                                 with the brief; validation has 50 queries

Each source holds four tables:
  queries.parquet    query_id, query, n_relevant, metric, earlier_w (the weights of the earlier run's statements), brief
                     (what the verifier was told with the brief: the instruction, the query and the LLM's background;
                     null when the set was only asked plainly. Asked plainly, it was told only "You are judging one
                     candidate image returned by an image search engine.")
  photos.parquet     query_id, rank, image_id, search_score (as the comparison used it), relevant, and the earlier run's
                     answers to its seven questions: earlier_h (whole query), earlier_s (statements), earlier_r (1-to-5).
                     The wide sources add the photo's rank in each earlier run's pool and the search score that run
                     stored, SigLIP2's similarity to the query in all four (rank_vith, score_vith, rank_siglip2,
                     score_siglip2, rank_xpool, score_xpool, rank_spool, score_spool; null when the photo was not in
                     that pool)
  questions.parquet  query_id, pass, question_id, family (W R C N M V X F), kind, text, target, options, levels,
                     required, decisive
  answers.parquet    query_id, pass, image_id, question_id, value (the answer on a 0-to-1 scale where 1 speaks for the
                     question's target), probs (each option's probability, for multiple choice and the 1-to-5 match).
                     The wide sources keep one row group per query

On INQUIRE-Rerank, the search score is SigLIP2's with the query lowercased, as in the paper. The other sets were asked
on earlier runs, so their seven questions and search scores come from those runs, not from the main tables.

    python export/export_bank.py --jevrank <jevrank> --inquire-full <inquire_full> --out <data>
    python export/export_bank.py --jevrank <jevrank> --inquire-full <inquire_full> --out <data> --only bank_inquire_wide_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from export_extras import sha256, write

RUNS = [  # source, bank folder, audit of the earlier run, its results (labels), lowercased search scores
    ("bank_inquire_rerank_test", "results/bank/inquire_test_sol", "J:audit/inquire_test", "results/inquire_test_jev.json", True),
    ("bank_inquire_full_vith_lc_val", "results/bank/inquire_fullval_sol", "F:audit/full_vith_lc_val_k500",
     "results/inquire_full/full_vith_lc_val_k500.json", False),
    ("bank_docci_s1", "results/bank/docci_s1_sol", "J:audit/docci_parts_maps_s1", "results/docci_parts_maps_s1_jev.json", False),
    ("bank_inquire_wide_test", "F:bank/inquire_wide_test_sol", "F:audit/full_wide_test", "results/inquire_full/full_vith_lc_test_k500.json", False),
    ("bank_inquire_wide_val", "F:bank/inquire_wide_val_sol", "F:audit/full_wide_val", "results/inquire_full/full_vith_lc_val_k500.json", False),
]
ARMS = ("vith", "siglip2", "xpool", "spool")       # the earlier runs whose checked photos make a wide pool
S = pa.string()
ANSWERS = pa.schema([("query_id", S), ("pass", S), ("image_id", S), ("question_id", S), ("value", pa.float64()),
                     ("probs", pa.list_(pa.float64()))])


def num(x):
    return None if x is None else float(x)


def at(J: Path, F: Path, p: str) -> Path:
    """J:... and plain paths are under the jevrank repository, F:... under the INQUIRE-Fullrank data folder."""
    return (F / p[2:]) if p.startswith("F:") else (J / (p[2:] if p.startswith("J:") else p))


def export(J: Path, F: Path, out: Path, src: str, bank: str, audit: str, results: str, lowercase: bool) -> list[dict]:
    bank_dir, adir = at(J, F, bank), at(J, F, audit)
    res = json.load(open(J / results, encoding="utf-8"))
    bench = res.get("benchmark", "")
    metric = res.get("headline", "")
    by_stem = {f"{bench}_{r['qid']}": r for r in res["per_query"]}
    prior = json.load(open(bank_dir / "prior_lowercase.json", encoding="utf-8")) if lowercase else {}
    stems: dict[str, dict] = {}
    for f in sorted((bank_dir / "answers").glob("*__*.json")):
        stem, name = f.stem.rsplit("__", 1)
        stems.setdefault(stem, {})[name] = f
    Qr, Pr, Kr, Ar = [], [], [], []
    wide = src.startswith("bank_inquire_wide_")
    fa = out / src / "answers.parquet"
    fa.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(fa, ANSWERS, compression="zstd") if wide else None
    n_answers = 0
    for stem, pf in stems.items():
        r = by_stem.get(stem) or next((v for k, v in by_stem.items() if stem.endswith("_" + k.split(bench + "_", 1)[-1])), None)
        af = adir / f"{stem}.json"
        if r is None or not af.exists():
            continue
        qid = stem[len(bench) + 1:] if stem.startswith(bench + "_") else stem
        A = json.load(open(af, encoding="utf-8"))
        passes = {name: json.load(open(f, encoding="utf-8")) for name, f in sorted(pf.items())}
        ids = [str(i) for i in next(iter(passes.values()))["ids"]]
        if any([str(i) for i in d["ids"]] != ids for d in passes.values()):
            continue
        sim = {str(c["image_id"]): float(c.get("sim") or 0) for c in A["candidates"]}
        arm = {str(c["image_id"]): c.get("src") or {} for c in A["candidates"]}
        if lowercase:
            low = prior.get(stem)
            if low and all(i in low for i in ids):
                sim = {i: float(low[i]) for i in ids}
        old = {str(x["image_id"]): (x.get("answers") or {}) for x in A.get("responses") or []}
        subs = [q for q in (A.get("questions") or []) if q["id"].startswith("sub")]
        rel = {str(x) for x in r["relevant"]}
        Qr.append(dict(query_id=qid, query=A["query"], n_relevant=len(rel), metric=metric,
                       earlier_w=[float(q.get("weight", 1.0)) for q in subs], brief=passes["brief"].get("state") if "brief" in passes else None))
        for k, i in enumerate(ids):
            a = old.get(i, {})
            row = dict(query_id=qid, rank=k + 1, image_id=i, search_score=sim.get(i, 0.0), relevant=i in rel,
                       earlier_h=num((a.get("holistic") or {}).get("prob")), earlier_r=num((a.get("relevance") or {}).get("expected")),
                       earlier_s=[num((a.get(q["id"]) or {}).get("prob")) for q in subs])
            if wide:
                for x in ARMS:
                    rs = arm.get(i, {}).get(x)
                    row[f"rank_{x}"], row[f"score_{x}"] = (int(rs[0]), float(rs[1])) if rs else (None, None)
            Pr.append(row)
        rows = []
        for name, d in passes.items():
            for q in d["questions"]:
                Kr.append(dict(query_id=qid, **{"pass": name}, question_id=q["id"], family=q["family"], kind=q["kind"], text=q["text"],
                               target=q.get("target"), options=list(q.get("options") or []) or None,
                               levels=[f"{a}: {b}" for a, b in q["levels"]] if q.get("levels") else None,
                               required=q.get("required"), decisive=q.get("decisive")))
                vals = d["answers"].get(q["id"]) or [None] * len(ids)
                probs = (d.get("probs") or {}).get(q["id"]) if q["kind"] != "noul" else None
                for k, i in enumerate(ids):
                    pr = probs[k] if probs is not None else None
                    rows.append(dict(query_id=qid, **{"pass": name}, image_id=i, question_id=q["id"], value=num(vals[k]),
                                     probs=None if pr is None else [float(x) for x in pr]))
        n_answers += len(rows)
        if writer is not None:                    # one row group per query: the wide sources are too large to hold
            writer.write_table(pa.Table.from_pylist(rows, schema=ANSWERS))
        else:
            Ar += rows
    if writer is not None:
        writer.close()
        ma = dict(file=fa.relative_to(out).as_posix(), rows=n_answers, bytes=fa.stat().st_size, sha256=sha256(fa))
    else:
        ma = write(Ar, ANSWERS, fa, out)
    photos = [("query_id", S), ("rank", pa.int32()), ("image_id", S), ("search_score", pa.float64()), ("relevant", pa.bool_()),
              ("earlier_h", pa.float64()), ("earlier_r", pa.float64()), ("earlier_s", pa.list_(pa.float64()))]
    if wide:
        photos += [c for x in ARMS for c in ((f"rank_{x}", pa.int32()), (f"score_{x}", pa.float64()))]
    m = [write(Qr, pa.schema([("query_id", S), ("query", S), ("n_relevant", pa.int32()), ("metric", S),
                              ("earlier_w", pa.list_(pa.float64())), ("brief", S)]), out / src / "queries.parquet", out),
         write(Pr, pa.schema(photos), out / src / "photos.parquet", out),
         write(Kr, pa.schema([("query_id", S), ("pass", S), ("question_id", S), ("family", S), ("kind", S), ("text", S), ("target", S),
                              ("options", pa.list_(S)), ("levels", pa.list_(S)), ("required", pa.bool_()), ("decisive", pa.bool_())]),
               out / src / "questions.parquet", out),
         ma]
    print(f"{src}: {len(Qr)} queries, {len(Pr):,} photos, {n_answers:,} answers")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jevrank", required=True, type=Path)
    ap.add_argument("--inquire-full", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=None, help="the sources to write (default: all)")
    a = ap.parse_args()
    mf = a.out / "manifest.json"
    for src, bank, audit, results, low in RUNS:
        if a.only and src not in a.only:
            continue
        entry = export(a.jevrank, a.inquire_full, a.out, src, bank, audit, results, low)
        manifest = json.load(open(mf, encoding="utf-8"))          # re-read: each source is saved as soon as it is done
        manifest[src] = entry
        with open(mf, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
