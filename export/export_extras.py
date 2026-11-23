#!/usr/bin/env python
"""Side tables that the answer tables alone do not determine, added to the data folder that export_answers.py wrote
(nothing it wrote is changed; manifest.json gains the new files).

  docci/targets.parquet                      the 1,000 DOCCI targets: query_id, row, image_id, DOCCI example id, the
                                             description's sentences and the random order they arrive in (Sec. 3, "One
                                             detail at a time"); n_sentences is what "sentences used" counts for "all"
  docci/images.parquet                       image_id (i<row>) -> DOCCI example id for the 5,000 test photos, so the
                                             figures can find a photo in DOCCI's own release
  docci_deep200_<level>/steered.parquet      the 100 photos the answers chose (Sec. 6, "Looking further"): the
                                             steered search's top new photos, in its order; candidates.parquet holds
                                             them, after the search's top 200
  inquire_full_<pool>_<split>/firststage.parquet
                                             the pool encoder's top 1,000 (image_id, score): Table 3's stop probability
                                             labels the 100 photos after the 500 checked, and Table 4 counts ranks 501-1,000
  costs/ledgers.parquet                      the request ledgers of the INQUIRE-Rerank test run: requests and tokens of the
                                             verifier and of the LLM that wrote the statements, with the prices used (Sec. 3)
  inquire_full_streams/streams.parquet       Table 4's feedback to the search: for each INQUIRE-Fullrank query, every saved
                                             SigLIP2 search over the 5 million photos (the query; each LLM-written text
                                             alone and anchored to the query; each family's texts and every text with the
                                             query as one vector; two searches fitted on the verifier's answers to the first
                                             photos), with its text and top 200, in the order the analysis read them
  inquire_taxa/taxa.parquet                  iNat24's category of every photo in the Fullrank first stages (the folder
                                             name: category id, kingdom, phylum, class, order, family, genus, species).
                                             Sec. 7 and Table 4 use it to tell a look-alike's species from the target's
  inquire_gpt4o_pool/pool.parquet            the photos GPT-4o reranks in the INQUIRE paper: CLIP ViT-H's top 100 for the
                                             query as written (INQUIRE's own first stage), in its order, with the iNat24
                                             path under which INQUIRE's authors cache GPT-4o's answers. With that cache,
                                             scripts/gpt4o_answers.py gives the paired comparison with GPT-4o (Sec. 4)

Only ids, ranks, scores, iNat24's relative image paths and DOCCI's own text are written: no local path, image or
configuration.

    python export/export_extras.py --jevrank <jevrank> --inquire-full <inquire_full> --docci-parts <docci_parts> --out <data>
    python export/export_extras.py --inquire-full <inquire_full> --out <data> --only gpt4o_pool
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(rows: list[dict], schema: pa.Schema, f: Path, root: Path) -> dict:
    f.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(t, f, compression="zstd")
    return dict(file=f.relative_to(root).as_posix(), rows=t.num_rows, bytes=f.stat().st_size, sha256=sha256(f))


def base(p: str) -> str:
    return os.path.splitext(p.replace("\\", "/").rsplit("/", 1)[-1])[0]


def docci(J: Path, P: Path, out: Path) -> list[dict]:
    pairs = json.load(open(J / "cache/grid/docci_test_pairs.json", encoding="utf-8"))
    ex = pairs.get("example_ids") or [base(p) for p in pairs["paths"]]
    files = [base(p) + ".jpg" for p in pairs["paths"]]
    img = [dict(image_id=f"i{k}", docci_example=str(ex[k]), file_name=files[k]) for k in range(len(files))]
    m = [write(img, pa.schema([("image_id", pa.string()), ("docci_example", pa.string()), ("file_name", pa.string())]),
               out / "docci/images.parquet", out)]
    L = json.load(open(P / "levels.json", encoding="utf-8"))["queries"]
    rows = []
    for qid, v in L.items():
        r = int(v["row"])
        rows.append(dict(query_id=qid, row=r, image_id=f"i{r}", docci_example=str(ex[r]), sentences=list(v["sentences"]),
                         arrival=[int(x) for x in v["arrival"]], n_sentences=len(v["sentences"])))
    m.append(write(rows, pa.schema([("query_id", pa.string()), ("row", pa.int32()), ("image_id", pa.string()),
                                    ("docci_example", pa.string()), ("sentences", pa.list_(pa.string())),
                                    ("arrival", pa.list_(pa.int32())), ("n_sentences", pa.int32())]),
                   out / "docci/targets.parquet", out))
    return m


def steered(P: Path, out: Path, manifest: dict) -> None:
    for lv in ("s1", "s2", "s3", "s4"):
        src = f"docci_deep200_{lv}"
        if not (out / src).exists():
            continue
        S = json.load(open(P / f"pools200s/steer__{lv}.json", encoding="utf-8"))
        rows = [dict(query_id=qid, image_id=str(i), steer_rank=k + 1) for qid, v in S["queries"].items() for k, i in enumerate(v["ids"])]
        e = write(rows, pa.schema([("query_id", pa.string()), ("image_id", pa.string()), ("steer_rank", pa.int32())]),
                  out / src / "steered.parquet", out)
        manifest[src] = [x for x in manifest.get(src, []) if x["file"] != e["file"]] + [e]


def firststage(F: Path, out: Path, manifest: dict) -> None:
    for pool in ("vith_lc", "siglip2_lc"):
        fs = json.load(open(F / f"firststage/{pool}.json", encoding="utf-8"))["queries"]
        for split in ("test", "val"):
            src = f"inquire_full_{pool}_{split}"
            if not (out / src).exists():
                continue
            qids = [r["query_id"] for r in pq.read_table(out / src / "queries.parquet", columns=["query_id"]).to_pylist()]
            rows = []
            for qid in qids:
                num = qid.split("_")[1]
                e = fs.get(num) or fs.get(str(int(num)))
                sig = e.get("sig") or [None] * len(e["ids"])
                rows += [dict(query_id=qid, rank=k + 1, image_id=str(i), score=float(s), siglip2_score=None if g is None else float(g))
                         for k, (i, s, g) in enumerate(zip(e["ids"][:1000], e["score"][:1000], sig[:1000]))]
            x = write(rows, pa.schema([("query_id", pa.string()), ("rank", pa.int32()), ("image_id", pa.string()),
                                       ("score", pa.float32()), ("siglip2_score", pa.float32())]), out / src / "firststage.parquet", out)
            manifest[src] = [y for y in manifest.get(src, []) if y["file"] != x["file"]] + [x]


def ledgers(J: Path, out: Path, manifest: dict) -> None:
    """Requests, tokens and prices of the INQUIRE-Rerank test run, for the cost per query of Sec. 3."""
    res = json.load(open(J / "results/inquire_test_jev.json", encoding="utf-8"))
    rows = []
    for role, key in (("verifier", "backend"), ("statements", "llm")):
        L = res["ledgers"][key]
        rows.append(dict(source="inquire_rerank_test", role=role, queries=int(res["n_queries"]), requests=int(L["requests"]),
                         input_tokens=int(L["input_tokens"]), output_tokens=int(L["output_tokens"]),
                         usd_per_million_input=float(L["usd_per_million_input"]), usd_per_million_output=float(L["usd_per_million_output"])))
    x = write(rows, pa.schema([("source", pa.string()), ("role", pa.string()), ("queries", pa.int32()), ("requests", pa.int64()),
                               ("input_tokens", pa.int64()), ("output_tokens", pa.int64()), ("usd_per_million_input", pa.float64()),
                               ("usd_per_million_output", pa.float64())]), out / "costs/ledgers.parquet", out)
    manifest["costs"] = [x]


def streams(F: Path, out: Path, manifest: dict) -> None:
    """The saved searches of xplore/search/streams.json (ids and texts only; the query's own ranking is firststage)."""
    S = json.load(open(F / "xplore/search/streams.json", encoding="utf-8"))["queries"]
    rows = []
    for pos, (num, q) in enumerate(S.items()):
        qid = f"{q['split']}_{int(num):03d}"
        for name, st in q["streams"].items():
            rows.append(dict(query_id=qid, position=pos, stream=name, text=st.get("text"), ids=[str(i) for i in st["ids"]]))
    x = write(rows, pa.schema([("query_id", pa.string()), ("position", pa.int32()), ("stream", pa.string()), ("text", pa.string()),
                               ("ids", pa.list_(pa.string()))]), out / "inquire_full_streams/streams.parquet", out)
    manifest["inquire_full_streams"] = [x]


def taxa(F: Path, out: Path, manifest: dict) -> None:
    """iNat24's category (the folder an image sits in) for every photo of the Fullrank first stages."""
    cat: dict[str, str] = {}
    for pool in ("vith_lc", "siglip2_lc", "vith"):
        p = F / f"firststage/{pool}.json"
        if not p.exists():
            continue
        for e in json.load(open(p, encoding="utf-8"))["queries"].values():
            for i, path in zip(e["ids"], e["paths"]):
                c = str(path).replace("\\", "/").split("/")[-2]
                if cat.setdefault(str(i), c) != c:
                    raise SystemExit(f"image {i} is in two categories: {cat[str(i)]} and {c}")
    rows = [dict(image_id=i, category=c) for i, c in sorted(cat.items())]
    x = write(rows, pa.schema([("image_id", pa.string()), ("category", pa.string())]), out / "inquire_taxa/taxa.parquet", out)
    manifest["inquire_taxa"] = [x]


def gpt4o_pool(F: Path, out: Path, manifest: dict) -> None:
    """INQUIRE's CLIP ViT-H top 100 per query (the query as written), as fed to GPT-4o by INQUIRE's two-stage script."""
    fs = json.load(open(F / "firststage/vith.json", encoding="utf-8"))["queries"]
    rows = []
    for num, e in sorted(fs.items(), key=lambda kv: int(kv[0])):
        qid = f"{e['split']}_{int(num):03d}"
        for k, (i, p) in enumerate(zip(e["ids"][:100], e["paths"][:100])):
            p = str(p).replace("\\", "/").lstrip("./")
            rows.append(dict(query_id=qid, inquire_query_id=int(num), split=e["split"], rank=k + 1, image_id=str(i),
                             image_path=p if p.lower().endswith(".jpg") else p + ".jpg"))
    src = "inquire_gpt4o_pool"
    x = write(rows, pa.schema([("query_id", pa.string()), ("inquire_query_id", pa.int32()), ("split", pa.string()),
                               ("rank", pa.int32()), ("image_id", pa.string()), ("image_path", pa.string())]),
              out / src / "pool.parquet", out)
    manifest[src] = [x]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jevrank", type=Path)
    ap.add_argument("--inquire-full", required=True, type=Path)
    ap.add_argument("--docci-parts", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=None, help="docci steered firststage ledgers streams taxa gpt4o_pool (default: all)")
    a = ap.parse_args()
    want = set(a.only or ("docci", "steered", "firststage", "ledgers", "streams", "taxa", "gpt4o_pool"))
    mf = a.out / "manifest.json"
    manifest = json.load(open(mf, encoding="utf-8"))
    if "docci" in want:
        manifest["docci"] = docci(a.jevrank, a.docci_parts, a.out)
    if "steered" in want:
        steered(a.docci_parts, a.out, manifest)
    if "firststage" in want:
        firststage(a.inquire_full, a.out, manifest)
    if "ledgers" in want:
        ledgers(a.jevrank, a.out, manifest)
    if "streams" in want:
        streams(a.inquire_full, a.out, manifest)
    if "taxa" in want:
        taxa(a.inquire_full, a.out, manifest)
    if "gpt4o_pool" in want:
        gpt4o_pool(a.inquire_full, a.out, manifest)
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    for k in ("docci", "docci_deep200_s1", "inquire_full_vith_lc_test", "costs", "inquire_full_streams", "inquire_taxa", "inquire_gpt4o_pool"):
        print(k, [(x["file"], x["rows"]) for x in manifest.get(k, [])][-2:])


if __name__ == "__main__":
    main()
