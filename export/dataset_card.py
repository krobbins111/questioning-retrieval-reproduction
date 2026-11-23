#!/usr/bin/env python
"""Write the dataset's README.md (the Hugging Face dataset card) into the data folder, from docs/DATASET_CARD.md, the
manifest and configs/release.json: the viewer's configs (one per kind of table, one split per source) and a table of
sizes. Run after export_answers.py and export_extras.py, before the upload.

    python export/dataset_card.py --out <data folder>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIND = {"queries": "queries", "questions": "questions", "candidates": "candidates", "answers": "answers",
        "firststage": "firststage", "steered": "steered", "taxa": "taxa", "pool": "gpt4o_pool", "streams": "streams", "ledgers": "ledgers",
        "searches": "steer_searches"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    man = json.loads((a.out / "manifest.json").read_text(encoding="utf-8"))
    rel = json.loads((ROOT / "configs/release.json").read_text(encoding="utf-8"))
    tmpl = (ROOT / "docs/DATASET_CARD.md").read_text(encoding="utf-8")
    order = sorted(man)
    by_kind: dict[str, list[tuple[str, str]]] = {}
    special = {"clef_flash/answers.parquet": "clef_flash", "winoground/answers.parquet": "winoground",
               "docci/targets.parquet": "docci_targets", "docci/images.parquet": "docci_images"}
    for src in order:
        for x in man[src]:
            if src.startswith("bank_"):                     # the larger question set: its own four tables
                stem = Path(x["file"]).stem                 # the wide pools' photos carry more columns: a config of their own
                kind = "bank_wide_photos" if stem == "photos" and src.startswith("bank_inquire_wide_") else "bank_" + stem
                by_kind.setdefault(kind, []).append((src, x["file"]))
                continue
            if x["file"] in special:
                by_kind.setdefault(special[x["file"]], []).append(("answers" if src in ("clef_flash", "winoground") else src, x["file"]))
                continue
            kind = KIND[Path(x["file"]).stem]
            by_kind.setdefault(kind, []).append((src, x["file"]))
    lines = []
    for kind in ("queries", "questions", "candidates", "answers", "clef_flash", "winoground", "docci_targets",
                 "docci_images", "firststage", "steered", "taxa", "gpt4o_pool", "streams", "steer_searches", "ledgers", "bank_queries",
                 "bank_photos", "bank_wide_photos", "bank_questions", "bank_answers"):
        if kind not in by_kind:
            continue
        lines.append(f"- config_name: {kind}")
        if kind == "queries":
            lines.append("  default: true")
        lines.append("  data_files:")
        for split, f in by_kind[kind]:
            split = split if kind not in ("docci_targets", "docci_images") else "docci"
            lines += [f"  - split: {split}", f"    path: {f}"]
    rows = lambda src, stem: next((x["rows"] for x in man[src] if Path(x["file"]).stem == stem), 0)  # noqa: E731
    jev = sum(rows(s, "answers") for s in order if s not in ("clef_flash", "winoground"))   # the bank's answers are Jev's too
    clef, wino = rows("clef_flash", "answers"), rows("winoground", "answers")
    nbytes = sum(x["bytes"] for v in man.values() for x in v)
    st = [f"{jev + clef + wino:,} answers in {sum(len(v) for v in man.values())} parquet files ({nbytes / 1e6:.0f} MB): "
          f"{jev:,} from Jev (`jev-1.14.0`), {clef:,} from Clef-Flash and {wino:,} on Winoground.", "",
          "| source | queries | photos checked | answers |", "|---|---:|---:|---:|"]
    for s in order:
        if s in ("docci", "inquire_gpt4o_pool", "inquire_taxa", "inquire_full_streams", "inquire_full_steer", "costs") or s.startswith("bank_"):
            continue
        q, c = rows(s, "queries"), rows(s, "candidates")
        st.append(f"| `{s}` | {f'{q:,}' if q else ''} | {f'{c:,}' if c else ''} | {rows(s, 'answers'):,} |")
    card = tmpl.replace("{{configs}}", "\n".join(lines)).replace("{{stats}}", "\n".join(st)).replace("{{repo_id}}", rel["repo_id"])
    (a.out / "README.md").write_text(card, encoding="utf-8")
    print(f"wrote {a.out / 'README.md'}: {len(by_kind)} configs; {jev + clef + wino:,} answers, {nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
