# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.8",
#     "torchvision>=0.23",
#     "transformers==5.10.2",
#     "accelerate",
#     "safetensors",
#     "pillow",
#     "numpy",
#     "pyarrow",
#     "huggingface_hub",
# ]
# ///
"""Clef-Flash, the paper's open verifier (Cloudflare/clef-flash on Hugging Face), on the released questions: it asks a
source's seven questions about the source's candidate photos and writes the answers in the release's format, so the
tables can be re-scored without the hosted API (Sec. 3).

Each photo goes in with the instruction "You are judging one candidate image returned by an image search engine. The
photo is attached." and the query's questions as typed decisions, worded as in the paper's run: the whole query as
"The photo shows: <query>.", each statement as "About the photo: <statement>" (yes or no), and the 1-to-5 match as
"About the photo: <question>" with the five level descriptions as its options. An answer is the softmax over the
question's options, as the model card computes it: p_yes for a yes-or-no question; for the match, p_levels and
expected, the mean level divided by 4 (Jev's scale).

Photos are not in the release. --images is the benchmark's own image folder:
  DOCCI     DOCCI's images folder; docci/images.parquet maps the release's photo ids to DOCCI's file names
  INQUIRE   the iNat24 photos, with --image-map: a CSV of image_id,path (path relative to --images), made from INQUIRE's
            metadata, which gives each photo's iNat24 file name (without --image-map, <image_id>.jpg is tried)
Photos are read as the paper's run read them: turned upright (EXIF), scaled down to 1,024 px on the long side and up to
280 px on the short one; DOCCI's photos were first saved at 1,024 px as JPEG (quality 90), as the run received them.
GPU arithmetic differs between machines, so expect the released answers back to a few decimal places, not exactly.

    python verifier/clef_flash.py --source inquire_rerank_test --images <iNat24> --image-map ids.csv --out rerank.parquet
    python verifier/clef_flash.py --source docci_s1 --images <DOCCI>/images --out docci_s1.parquet --compare
    python verifier/clef_flash.py --source docci_s1 --images <DOCCI>/images --limit 2 --fake-model --out t.parquet

Needs a CUDA GPU (the paper's run: one model copy per 24 GB card), and the packages above (uv run reads them from this
file's header). The model's own loader, joint_schema_model.py, comes with the model's files. Answers are saved as they
come (<out>.jsonl beside the output), so a stopped run picks up where it was; the parquet is written at the end.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qr.data import data_root  # noqa: E402

STATE = "You are judging one candidate image returned by an image search engine. The photo is attached."
# the release's Clef-Flash answers name each benchmark set as the run did
CLEF_SET = {"inquire_rerank_test": "inquire_rerank", "inquire_full_vith_lc_test": "inquire_full", "inquire_full_siglip2_lc_test": "inquire_full",
            "docci_all": "docci_all", "docci_s1": "docci_s1", "docci_s2": "docci_s2", "docci_s3": "docci_s3", "docci_s4": "docci_s4"}
SCHEMA = pa.schema([("query_id", pa.string()), ("image_id", pa.string()), ("question_id", pa.string()), ("p_yes", pa.float64()),
                    ("p_levels", pa.list_(pa.float64())), ("expected", pa.float64()), ("verifier", pa.string()), ("error", pa.bool_())])


def payload(kind: str, text: str, levels) -> dict:
    """A released question as Clef-Flash's typed decision, worded as the paper's run worded it."""
    text = text.strip()
    if kind == "noul":
        low = text.lower()
        if low.startswith("this image shows:"):
            text = "The photo shows: " + text[len("this image shows:"):].strip()
        elif low.startswith("the image shows"):
            text = "The photo shows" + text[len("the image shows"):]
        else:
            text = f"About the photo: {text}"
        return {"type": "noul", "instructions": text}
    if kind == "score":
        return {"type": "score", "instructions": "About the photo: " + text, "criteria": [d for _, d in levels]}
    raise ValueError(f"no wording for a question of kind {kind!r}")


def requests(source: str, top: int, limit: int | None) -> list[tuple[str, str, dict]]:
    """(query, photo, questions) for the first `top` candidates of each query, in search order."""
    d = data_root() / source
    K = defaultdict(list)
    for r in pq.read_table(d / "questions.parquet").to_pylist():
        K[r["query_id"]].append(r)
    C = defaultdict(list)
    for r in pq.read_table(d / "candidates.parquet", columns=["query_id", "image_id", "search_rank"]).to_pylist():
        C[r["query_id"]].append(r)
    out = []
    for q in sorted(K)[:limit]:
        qs = {x["question_id"]: payload(x["kind"], x["text"], json.loads(x["levels"]) if x["levels"] else None)
              for x in K[q] if x["role"] != "rewrite"}
        for c in sorted(C[q], key=lambda r: r["search_rank"])[:top]:
            out.append((q, c["image_id"], qs))
    return out


def photo_paths(source: str, images: Path, image_map: Path | None):
    """image id -> file, and whether the run re-saved the photo (DOCCI) before reading it."""
    if source.startswith("docci"):
        t = pq.read_table(data_root() / "docci/images.parquet", columns=["image_id", "file_name"]).to_pydict()
        names = dict(zip(t["image_id"], t["file_name"]))
        return (lambda i: images / names[i]), True
    if image_map:
        with open(image_map, newline="", encoding="utf-8") as fh:
            names = {r["image_id"]: r["path"] for r in csv.DictReader(fh)}
        return (lambda i: images / names[i]), False
    return (lambda i: images / f"{i}.jpg"), False


def read_photo(p: Path, resave: bool, max_side: int = 1024, min_side: int = 280):
    from PIL import Image, ImageOps
    img = Image.open(p)
    if resave:                                   # as the photos were packed for the run
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        buf.seek(0)
        img = Image.open(buf)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    if min(img.size) < min_side:
        s = min_side / min(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BILINEAR)
    return img


def rows_of(q: str, i: str, ans: dict) -> list[dict]:
    """The release's format: p_yes for yes or no, p_levels and expected (mean level / 4) for the match."""
    out = []
    for qn, (opts, probs) in ans.items():
        if list(opts[:2]) == ["true", "false"]:
            out.append(dict(query_id=q, image_id=i, question_id=qn, p_yes=float(probs[0]), p_levels=None, expected=None))
        else:
            p = [float(v) for v in probs]
            out.append(dict(query_id=q, image_id=i, question_id=qn, p_yes=None, p_levels=p, expected=sum(k * v for k, v in enumerate(p)) / max(1, len(p) - 1)))
    return out


def run(a) -> list[dict]:
    reqs = requests(a.source, a.top, a.limit)
    side = Path(str(a.out) + ".jsonl")
    done = {}
    if side.exists():
        for line in open(side, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(d["q"], d["i"])] = d
    todo = [r for r in reqs if (r[0], r[1]) not in done]
    print(f"{a.source}: {len(reqs):,} photos to answer, {len(reqs) - len(todo):,} answered before", flush=True)
    path, resave = photo_paths(a.source, Path(a.images), Path(a.image_map) if a.image_map else None)
    if a.fake_model:
        M = model = proc0 = None
    else:
        import torch
        from huggingface_hub import snapshot_download
        mpath = snapshot_download(a.model, revision=a.revision)
        sys.path.insert(0, mpath)
        import joint_schema_model as M  # noqa: E402  (ships with the model)
        from transformers import AutoProcessor
        model, proc0 = M.load_release_model(mpath, device=a.device)
        model.eval()
        pad, dev = proc0.tokenizer.pad_token_id, torch.device(a.device)
        local = threading.local()

        def processor():
            if not hasattr(local, "proc"):
                local.proc = AutoProcessor.from_pretrained(mpath)   # one per thread: fast tokenizers are not thread-safe
            return local.proc

    def encode(r):
        q, i, qs = r
        try:
            rec = {"id": f"{q}|{i}", "state": STATE, "questions": qs, "images": [read_photo(path(i), resave)]}
            if a.fake_model:
                return rec
            p = processor()
            return M.encode_record(p.tokenizer, rec, processor=p)
        except Exception as e:  # noqa: BLE001
            return e
    fh = open(side, "a", encoding="utf-8")

    def emit(r, enc, logits, err=None):
        q, i, qs = r
        if err is not None:
            d = {"q": q, "i": i, "err": err[:400]}
        elif a.fake_model:
            ans = {}
            for qid, x in qs.items():
                opts = ["true", "false"] if x["type"] == "noul" else [str(k) for k in range(len(x["criteria"]))]
                v = [random.random() for _ in opts]
                ans[qid] = [opts, [round(t / sum(v), 6) for t in v]]
            d = {"q": q, "i": i, "a": ans}
        else:
            d = {"q": q, "i": i, "a": {qm.question_id: [list(qm.option_ids), [round(v, 6) for v in lg.float().softmax(-1).tolist()]]
                                       for qm, lg in zip(enc.questions, logits)}}
        done[(q, i)] = d
        fh.write(json.dumps(d) + "\n")
    B = a.batch

    def answer(items):
        nonlocal B
        if a.fake_model:
            for r, e in items:
                emit(r, e, None)
            return
        import torch
        try:
            batch = M.collate_records([e for _, e in items], pad, dev)
            with torch.inference_mode():
                outs = model(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(items) == 1:
                emit(items[0][0], None, None, "out of memory")
                return
            B = max(1, min(B, len(items) // 2))
            answer(items[:len(items) // 2]); answer(items[len(items) // 2:])
            return
        except Exception as e:  # noqa: BLE001
            if len(items) == 1:
                emit(items[0][0], None, None, f"forward: {type(e).__name__}: {e}")
                return
            answer(items[:len(items) // 2]); answer(items[len(items) // 2:])
            return
        for (r, e), lg in zip(items, outs):
            emit(r, e, lg)
    pool = ThreadPoolExecutor(max_workers=a.threads)

    def start(j):
        chunk = todo[j:j + B]
        return chunk, [pool.submit(encode, r) for r in chunk]
    t0, n = time.time(), 0
    cur = start(0)
    j = len(cur[0])
    while cur[0]:
        nxt = start(j)                           # the next photos are read while the GPU answers these
        j += len(nxt[0])
        chunk, futs = cur
        items = []
        for r, f in zip(chunk, futs):
            e = f.result()
            if isinstance(e, Exception):
                emit(r, None, None, f"encode: {type(e).__name__}: {e}")
            else:
                items.append((r, e))
        if items:
            answer(items)
        fh.flush()
        n += len(chunk)
        if n % (25 * a.batch) < len(chunk) or not nxt[0]:
            print(f"  {n:,}/{len(todo):,} photos, {n / max(1e-9, time.time() - t0):.1f} a second", flush=True)
        cur = nxt
    fh.close()
    rows = []
    for q, i, _ in reqs:
        d = done.get((q, i))
        if d is None or "err" in d:
            rows.append(dict(query_id=q, image_id=i, question_id=None, p_yes=None, p_levels=None, expected=None, error=True))
            continue
        rows += [dict(r, error=False) for r in rows_of(q, i, d["a"])]
    for r in rows:
        r["verifier"] = "clef-flash" + (" (fake)" if a.fake_model else "")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), a.out, compression="zstd")
    print(f"wrote {len(rows):,} answers to {a.out} ({sum(r['error'] for r in rows)} photos without an answer)", flush=True)
    return rows


def compare(source: str, rows: list[dict]) -> None:
    """How far the new answers are from the released Clef-Flash answers to the same questions about the same photos."""
    want = CLEF_SET.get(source)
    if want is None:
        print(f"the release has no Clef-Flash answers for {source}")
        return
    t = pq.read_table(data_root() / "clef_flash/answers.parquet", columns=["source", "query_id", "image_id", "question_id", "p_yes", "expected"],
                      filters=[("source", "=", want)]).to_pydict()
    old = {(q, i, k): (p, e) for q, i, k, p, e in zip(t["query_id"], t["image_id"], t["question_id"], t["p_yes"], t["expected"])}
    dy, de = [], []
    for r in rows:
        o = old.get((r["query_id"], r["image_id"], r["question_id"]))
        if o is None or r["error"]:
            continue
        if r["p_yes"] is not None and o[0] is not None:
            dy.append(abs(r["p_yes"] - o[0]))
        if r["expected"] is not None and o[1] is not None:
            de.append(abs(r["expected"] - o[1]))
    for name, v in (("yes-or-no answers (p_yes)", dy), ("1-to-5 match (expected)", de)):
        if v:
            v = sorted(v)
            print(f"{name}: {len(v):,} matched, mean difference {sum(v) / len(v):.4f}, median {v[len(v) // 2]:.4f}, largest {v[-1]:.4f}")
        else:
            print(f"{name}: nothing to compare")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="a source of the release: inquire_rerank_test, inquire_full_vith_lc_test, docci_s1, ...")
    ap.add_argument("--images", required=True, help="the benchmark's image folder")
    ap.add_argument("--image-map", default=None, help="INQUIRE: CSV with columns image_id,path (path relative to --images)")
    ap.add_argument("--out", required=True, help="the answers, as a parquet file")
    ap.add_argument("--top", type=int, default=100, help="candidates per query, in search order (the paper's run: 100)")
    ap.add_argument("--limit", type=int, default=None, help="the first N queries only")
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--revision", default=None, help="a revision of the model's repository")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=5, help="threads that read and encode the photos")
    ap.add_argument("--compare", action="store_true", help="report the difference to the released Clef-Flash answers")
    ap.add_argument("--fake-model", action="store_true", help="random answers, no GPU: tests the reading and writing")
    a = ap.parse_args()
    rows = run(a)
    if a.compare:
        compare(a.source, rows)


if __name__ == "__main__":
    main()
