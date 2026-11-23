#!/usr/bin/env python
"""Export the saved verifier answers behind the paper into compact parquet tables for the Hugging Face dataset.

Reads the run files and per-query answer files written by the experiment code (and the Clef-Flash job's answers), keeps
only what re-scoring needs, and writes one folder per source under OUT:

  queries.parquet     source, benchmark, subset, split, query_id, query, relevant (image ids; DOCCI: the one target)
  questions.parquet   query_id, question_id, role (holistic | sub | relevance), kind (noul | score), text, weight,
                      levels (rating questions: JSON list of [label, meaning])
  candidates.parquet  query_id, image_id, search_rank, search_score, and siglip2_score where the pool is CLIP ViT-H's
  answers.parquet     query_id, image_id, question_id, p_yes (yes/no questions), p_levels (the 1-to-5 rating, five
                      probabilities), expected (mean rating scaled to [0, 1]), verifier, input_tokens, error

Nothing else is copied: no image paths or other local paths, timings, raw API payloads or run configuration. Image ids
are the benchmarks' own. A manifest (rows and sha256 per file) is written next to the tables.

  python export_answers.py --jevrank <jevrank dir> --inquire-full <inquire_full data dir> --out <dir> [--only NAME ...]
  python export_answers.py ... --list          # the sources and whether their files exist
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LEVELS = ("s1", "s2", "s3", "s4", "all")
META = ("query_id", "supercategory", "category", "iconic_group")   # INQUIRE's own query fields; never paths
QID = re.compile(r"((?:test|val)_\d+|docci\d+)$")


def sources(J: Path, F: Path) -> list[dict]:
    """Every run whose answers the paper uses. J = jevrank folder, F = inquire_full data folder."""
    out = []
    for lv in LEVELS:   # DOCCI, sentences arriving one by one (Tables 1-3, Figures 1, 2, 4, 5)
        out.append(dict(name=f"docci_{lv}", benchmark="docci", subset=lv, split="test",
                        run=J / f"results/docci_parts_frozen_{lv}_jev.json", audit=J / f"audit/docci_parts_frozen_{lv}"))
    # DOCCI, 500 more targets with full descriptions, 200 candidates, one 64-token embedding (Table 2, held-out row):
    # the run holds 1,000 descriptions; the 500 that are also the main targets are dropped here.
    out.append(dict(name="docci_heldout", benchmark="docci", subset="heldout_full", split="test",
                    run=J / "results/grid_docci_raw_jev_k200_q1000.json", audit=J / "audit/grid_docci_raw_q1000",
                    exclude_run=J / "results/docci_parts_frozen_all_jev.json"))
    for lv in LEVELS[:4]:   # DOCCI with the search's top 200 checked, plus verifier-guided additions (Sec. 6, Table 3)
        out.append(dict(name=f"docci_deep200_{lv}", benchmark="docci", subset=f"deep200_{lv}", split="test",
                        run=J / f"results/docci_parts_frozen_deep200_{lv}_jev.json",
                        audit=J / f"audit/docci_parts_frozen_deep200_{lv}"))
    # INQUIRE-Rerank, 160 test queries, the fixed 100 photos (Table 1). The search score is SigLIP2 on the lowercased
    # query (Sec. 3); the score on the query as written is kept in siglip2_score.
    out.append(dict(name="inquire_rerank_test", benchmark="inquire_rerank", subset="rerank", split="test",
                    run=J / "results/inquire_test_jev.json", audit=J / "audit/inquire_test",
                    prior=J / "results/bank/inquire_test_terra/prior_lowercase.json"))
    # INQUIRE-Rerank, 40 validation queries (score weights fitted on validation, supplement)
    out.append(dict(name="inquire_rerank_val", benchmark="inquire_rerank", subset="rerank", split="val",
                    run=J / "results/dev_s0_inquire_val.json", audit=J / "audit/dev_s0_inquire_val"))
    # INQUIRE-Fullrank, two pools of 500 photos; 100 and 200 are prefixes of these (Table 1, Figure 3, Sec. 4-6)
    for pool in ("vith_lc", "siglip2_lc"):
        for split in ("test", "val"):
            out.append(dict(name=f"inquire_full_{pool}_{split}", benchmark="inquire_full", subset=pool, split=split,
                            run=J / f"results/final/inquire_full/full7_{pool}_{split}_k500.json",
                            audit=F / f"audit/final7_full7_{pool}_{split}_k500", firststage=F / f"firststage/{pool}.json"))
    return out


# ---------------------------------------------------------------------------------------------------------------- utils
def jload(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(tables: dict[str, dict[str, list]], schema: dict[str, pa.Schema], out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, cols in tables.items():
        t = pa.Table.from_pydict(cols, schema=schema[name])
        f = out / f"{name}.parquet"
        pq.write_table(t, f, compression="zstd")
        manifest.append(dict(file=f"{out.name}/{f.name}", rows=t.num_rows, bytes=f.stat().st_size, sha256=sha256(f)))
    return manifest


SCHEMA = {
    "queries": pa.schema([("source", pa.string()), ("benchmark", pa.string()), ("subset", pa.string()),
                          ("split", pa.string()), ("query_id", pa.string()), ("query", pa.string()),
                          ("relevant", pa.list_(pa.string())), ("meta", pa.string())]),
    "questions": pa.schema([("query_id", pa.string()), ("question_id", pa.string()), ("role", pa.string()),
                            ("kind", pa.string()), ("text", pa.string()), ("weight", pa.float32()),
                            ("levels", pa.string())]),
    "candidates": pa.schema([("query_id", pa.string()), ("image_id", pa.string()), ("search_rank", pa.int32()),
                             ("search_score", pa.float32()), ("siglip2_score", pa.float32())]),
    "answers": pa.schema([("query_id", pa.string()), ("image_id", pa.string()), ("question_id", pa.string()),
                          ("p_yes", pa.float32()), ("p_levels", pa.list_(pa.float32())), ("expected", pa.float32()),
                          ("verifier", pa.string()), ("input_tokens", pa.int32()), ("error", pa.bool_())]),
}


def empty() -> dict[str, dict[str, list]]:
    return {n: {f.name: [] for f in s} for n, s in SCHEMA.items()}


# ------------------------------------------------------------------------------------------------ jevrank runs (Jev)
def export_run(src: dict, out_root: Path) -> list[dict]:
    run = jload(src["run"])
    name = run["benchmark"]
    skip = set()
    if src.get("exclude_run"):
        skip = {r["qid"] for r in jload(src["exclude_run"])["per_query"]}
    prior = jload(src["prior"]) if src.get("prior") else None
    first = None
    if src.get("firststage"):   # the pool's own encoder: ranks and similarities over the 5M photos
        first = jload(src["firststage"])["queries"]
    T = empty()
    n_q = n_missing = 0
    for row in run["per_query"]:
        qid = row["qid"]
        if qid in skip:
            continue
        f = Path(src["audit"]) / f"{name}_{qid}.json"
        if not f.exists():
            n_missing += 1
            continue
        A = jload(f)
        n_q += 1
        q = T["queries"]
        for k, v in (("source", src["name"]), ("benchmark", src["benchmark"]), ("subset", src["subset"]),
                     ("split", src["split"]), ("query_id", qid), ("query", A["query"]),
                     ("relevant", [str(x) for x in row.get("relevant", [])]),
                     ("meta", json.dumps({k2: v2 for k2, v2 in (row.get("meta") or {}).items() if k2 in META}) or None)):
            q[k].append(v)
        qs = A["questions"]
        for x in qs:
            T["questions"]["query_id"].append(qid)
            T["questions"]["question_id"].append(x["id"])
            T["questions"]["role"].append(x.get("role"))
            T["questions"]["kind"].append(x.get("kind"))
            T["questions"]["text"].append(x.get("text"))
            T["questions"]["weight"].append(float(x.get("weight", 1.0)))
            T["questions"]["levels"].append(json.dumps(x.get("levels")) if x.get("levels") else None)
        own = None
        if first is not None:
            num = str(row["meta"].get("query_id"))
            e = first.get(num) or first.get(str(int(num)))
            own = {str(i): (j + 1, float(s)) for j, (i, s) in enumerate(zip(e["ids"], e["score"]))}
        for c in A["candidates"]:
            iid = str(c["image_id"])
            T["candidates"]["query_id"].append(qid)
            T["candidates"]["image_id"].append(iid)
            if own is not None:   # CLIP ViT-H pool: its own rank and score; SigLIP2's score kept as a second column
                rk, sc = own[iid]
                T["candidates"]["search_rank"].append(rk)
                T["candidates"]["search_score"].append(sc)
                T["candidates"]["siglip2_score"].append(float(c["sim"]) if src["subset"].startswith("vith") else None)
            elif prior is not None:   # lowercased-query score as the search score, the as-written score kept
                T["candidates"]["search_rank"].append(int(c["rank_embedding"]))
                T["candidates"]["search_score"].append(float(prior[f"{name}_{qid}"][iid]))
                T["candidates"]["siglip2_score"].append(float(c["sim"]))
            else:
                T["candidates"]["search_rank"].append(int(c["rank_embedding"]))
                T["candidates"]["search_score"].append(float(c["sim"]))
                T["candidates"]["siglip2_score"].append(None)
        levels = {x["id"]: [l[0] for l in x.get("levels") or []] for x in qs}
        for r in A["responses"]:
            iid = str(r["image_id"])
            usage = r.get("usage") or {}
            tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
            for x in qs:
                a = (r.get("answers") or {}).get(x["id"])
                T["answers"]["query_id"].append(qid)
                T["answers"]["image_id"].append(iid)
                T["answers"]["question_id"].append(x["id"])
                T["answers"]["verifier"].append(r.get("model") or "jev")
                T["answers"]["input_tokens"].append(int(tokens) if tokens is not None else None)
                T["answers"]["error"].append(a is None or bool(r.get("error")))
                if a is None:
                    T["answers"]["p_yes"].append(None); T["answers"]["p_levels"].append(None); T["answers"]["expected"].append(None)
                elif x.get("kind") == "score":
                    probs = a.get("probs") or {}
                    T["answers"]["p_yes"].append(None)
                    T["answers"]["p_levels"].append([float(probs.get(l, 0.0)) for l in levels[x["id"]]])
                    T["answers"]["expected"].append(None if a.get("expected") is None else float(a["expected"]))
                else:
                    T["answers"]["p_yes"].append(None if a.get("prob") is None else float(a["prob"]))
                    T["answers"]["p_levels"].append(None); T["answers"]["expected"].append(None)
    man = write(T, SCHEMA, out_root / src["name"])
    print(f"{src['name']}: {n_q} queries ({n_missing} without an answer file); "
          + ", ".join(f"{m['file'].split('/')[-1]} {m['rows']:,} rows {m['bytes'] / 1e6:.1f} MB" for m in man))
    return man


# ------------------------------------------------------------------------------------------- Clef-Flash (open model)
def export_clef(F: Path, out_root: Path) -> list[dict]:
    """The Clef-Flash job's answers: one row per (query, photo, question), keyed like the Jev tables. Clef answers the
    rating on five levels '0'-'4'; expected = mean level / 4, the same scale as Jev's."""
    cdir = F / "clef"
    keys = {}
    for line in open(cdir / "in/queries.jsonl", encoding="utf-8"):
        d = json.loads(line)
        keys[d["q"]] = d
    T = {"answers": {f.name: [] for f in SCHEMA["answers"]}}
    T["answers"]["source"] = []; T["answers"]["query_key"] = []
    seen = set()
    for fn in sorted(glob.glob(str(cdir / "out/answers_*.jsonl"))):
        for line in open(fn, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "a" not in d or d["rid"] in seen:
                continue
            seen.add(d["rid"])
            ds, qkey, iid = d["rid"].split("|")
            m = QID.search(qkey)          # the run's own query id at the end of the key: test_000, val_012, docci1005
            qid = m.group(1) if m else qkey
            for qn, (opts, probs) in d["a"].items():
                A = T["answers"]
                A["source"].append(ds); A["query_key"].append(qkey); A["query_id"].append(qid); A["image_id"].append(iid); A["question_id"].append(qn)
                A["verifier"].append("clef-flash"); A["input_tokens"].append(None); A["error"].append(False)
                if opts[:2] == ["true", "false"]:
                    A["p_yes"].append(float(probs[0])); A["p_levels"].append(None); A["expected"].append(None)
                else:
                    p = [float(v) for v in probs]
                    A["p_yes"].append(None); A["p_levels"].append(p)
                    A["expected"].append(sum(i * v for i, v in enumerate(p)) / max(1, len(p) - 1))
    schema = {"answers": SCHEMA["answers"].append(pa.field("source", pa.string())).append(pa.field("query_key", pa.string()))}
    man = write(T, schema, out_root / "clef_flash")
    print(f"clef_flash: {len(seen):,} photo answers; " + ", ".join(f"{m['rows']:,} rows {m['bytes'] / 1e6:.1f} MB" for m in man))
    return man


# ------------------------------------------------------------------------------------------------------- Winoground
def export_winoground(J: Path, out_root: Path) -> list[dict]:
    """One yes/no question per (example, caption, image): the run and its two controls (no image; captions shuffled
    across examples). label = 1 where the caption belongs to the image. Example ids are Winoground's own; no caption,
    image or Winoground annotation (tags) is written, since Winoground is shared under its own research-only terms."""
    cols = {k: [] for k in ("mode", "example", "caption", "image", "p_yes", "label")}
    for mode in ("single", "textonly", "shuffled"):
        f = J / f"results/winoground_{mode}.json"
        if not f.exists():
            continue
        for r in jload(f)["records"]:
            for k, v in (("mode", mode), ("example", str(r["example"])), ("caption", int(r["caption"])),
                         ("image", int(r["image"])), ("p_yes", None if r.get("p_yes") is None else float(r["p_yes"])),
                         ("label", int(r["label"]))):
                cols[k].append(v)
    schema = pa.schema([("mode", pa.string()), ("example", pa.string()), ("caption", pa.int8()), ("image", pa.int8()),
                        ("p_yes", pa.float32()), ("label", pa.int8())])
    man = write({"answers": cols}, {"answers": schema}, out_root / "winoground")
    print(f"winoground: {len(cols['mode']):,} rows")
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jevrank", required=True, type=Path)
    ap.add_argument("--inquire-full", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", help="source names (see --list); 'clef_flash' for the open model's answers")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    S = sources(a.jevrank, a.inquire_full)
    if a.list:
        for s in S:
            print(f"{s['name']:<30} run {'ok' if s['run'].exists() else 'MISSING'} | answers {'ok' if Path(s['audit']).exists() else 'MISSING'}")
        print(f"{'clef_flash':<30} answers {'ok' if (a.inquire_full / 'clef/out').exists() else 'MISSING'}")
        print(f"{'winoground':<30} answers {'ok' if (a.jevrank / 'results/winoground_single.json').exists() else 'MISSING'}")
        return
    want = set(a.only or [s["name"] for s in S] + ["clef_flash", "winoground"])
    manifest_f = a.out / "manifest.json"
    manifest = jload(manifest_f) if manifest_f.exists() else {}
    for s in S:
        if s["name"] in want:
            t0 = time.time()
            manifest[s["name"]] = export_run(s, a.out)
            print(f"   {time.time() - t0:.0f} s")
    if "clef_flash" in want:
        manifest["clef_flash"] = export_clef(a.inquire_full, a.out)
    if "winoground" in want:
        manifest["winoground"] = export_winoground(a.jevrank, a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(manifest_f, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
