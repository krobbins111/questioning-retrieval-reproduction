#!/usr/bin/env python
"""The supplement's example figures, from the released answers and DOCCI's own photos.

  example_anole.pdf        Fig. S4: one DOCCI description (docci4982, all sentences), the search's top three photos,
                           their similarity and verifier score, and the probability of yes to three of the statements
  example_lookalikes.pdf   Fig. S5: six DOCCI targets with the photo each is most easily confused with, and the
                           probability of yes to each of the seven questions

Every number drawn is computed here from the answer tables (paper_numbers.py, group `examples`, checks the ones the
captions quote). The photos are DOCCI's test images (CC BY 4.0), by the file names in docci/images.parquet.

    python paper/examples.py --images <folder with DOCCI's test images>
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap               # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle            # noqa: E402
from PIL import Image, ImageOps                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qr.data import data_root, load, relevant                       # noqa: E402
from qr.score import lead, rank                                     # noqa: E402

OUT = ROOT / "paper" / "out" / "figures"
TAU = 0.114
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED, FAINT, RULE = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
BLUE, ORANGE = "#1c5cab", "#eb6834"
RAMP = LinearSegmentedColormap.from_list("pyes", ["#f3f6fb", "#c9dbf3", "#86b6ef", "#2a78d6", "#104281"])
LV = {"all": "All sentences", "s2": "Two sentences", "s3": "Three sentences", "s4": "Four sentences"}
# (level, target, which photo to set beside it): five look-alikes that search ranks first, one the verifier ranks first
LOOKALIKES = [("all", "docci3262", "search"), ("all", "docci72", "search"), ("all", "docci3048", "search"),
              ("s2", "docci4470", "search"), ("s3", "docci3430", "search"), ("all", "docci753", "verifier")]
ANOLE = ("docci4982", ("sub0", "sub1", "sub4"))


def questions(source: str) -> dict:
    t = pq.read_table(data_root() / source / "questions.parquet", columns=["query_id", "question_id", "text"]).to_pydict()
    out: dict = {}
    for q, i, x in zip(t["query_id"], t["question_id"], t["text"]):
        out.setdefault(q, {})[i] = x
    return out


def answer(p, qid: str):
    if qid == "holistic":
        return p.h
    if qid == "relevance":
        return p.r
    return p.s[int(qid[3:])]


def example(lv: str, t: str, beside: str, texts: dict) -> dict:
    Q, P = load(f"docci_{lv}")
    R = rank(P[t])
    f = {p.image_id: (k + 1, s) for k, (p, s) in enumerate(R)}
    rel = relevant(Q[t])
    by_search = sorted(P[t], key=lambda p: (-p.search_score, p.search_rank))
    gt = next(p for p in P[t] if p.image_id in rel)
    other = next(p for p in (by_search if beside == "search" else [p for p, _ in R]) if p.image_id not in rel)
    qids = ["holistic"] + [f"sub{j}" for j in range(len(gt.s))] + ["relevance"]
    rows = [dict(id=i, text=texts[t][i], p=[answer(gt, i), answer(other, i)]) for i in qids]
    return dict(lv=lv, t=t, kind="ok" if R[0][0].image_id in rel else "fail", query=Q[t]["query"], gt=gt.image_id,
                other=other.image_id, sim=[gt.search_score, other.search_score], srank=[gt.search_rank, other.search_rank],
                final=[f[gt.image_id][1], f[other.image_id][1]], vrank=[f[gt.image_id][0], f[other.image_id][0]], lead=lead(R), rows=rows)


def photo(images: Path, files: dict, iid: str, size: int = 400) -> np.ndarray:
    im = Image.open(images / files[iid]).convert("RGB")
    return np.asarray(ImageOps.fit(im, (size, size), method=Image.LANCZOS))


def lookalikes(images: Path, files: dict) -> None:
    texts = {lv: questions(f"docci_{lv}") for lv in {lv for lv, _, _ in LOOKALIKES}}
    EX = [example(lv, t, b, texts[lv]) for lv, t, b in LOOKALIKES]
    W = 6.875; PW = 3.33; PGAP = W - 2 * PW - 0.08
    LABW, P, CG, FS = 1.40, 0.82, 0.06, 5.6

    def label(q):
        if q["id"] == "holistic":
            return ["The whole query"]
        if q["id"] == "relevance":
            return ["Match score, 1 to 5"]
        return textwrap.wrap(q["text"], 33)

    def rows_h(e):
        return sum(0.112 + 0.083 * (len(label(q)) - 1) for q in e["rows"])

    def panel_h(e):
        return 0.28 + P + 0.15 + rows_h(e) + 0.20

    def first_sentence(text, n=165):
        t = text.strip(); s = t.split(". ")[0].rstrip(".") + "."
        if len(s) > n:
            s = s[:n].rsplit(" ", 1)[0] + "…"
        elif len(s) < len(t):
            s += " …"
        return s

    rowsH = [max(panel_h(EX[i]), panel_h(EX[i + 1])) for i in range(0, len(EX), 2)]
    H = sum(rowsH) + 0.14 * (len(rowsH) - 1) + 0.06
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ytop = H - 0.03
    for r in range(len(rowsH)):
        for c in range(2):
            e = EX[2 * r + c]; px = 0.04 + c * (PW + PGAP); y = ytop
            ok = e["kind"] == "ok"
            ax.text(px, y - 0.07, LV[e["lv"]], color=INK, fontsize=6.6, fontweight="bold", va="center")
            sure = e["lead"] >= TAU
            chip = "sure, answers: right" if (sure and ok) else "not sure (lead %.2f): asks" % e["lead"]
            cw = 1.30 if ok else 1.45
            ax.add_patch(FancyBboxPatch((px + PW - cw, y - 0.14), cw, 0.14, boxstyle="round,pad=0,rounding_size=0.04",
                                        fc=BLUE if (sure and ok) else "white", ec=BLUE if (sure and ok) else ORANGE, lw=0.9))
            ax.text(px + PW - cw / 2, y - 0.07, chip, color="white" if (sure and ok) else ORANGE, fontsize=5.8, fontweight="bold",
                    ha="center", va="center")
            ax.text(px, y - 0.30, "\n".join(textwrap.wrap("“" + first_sentence(e["query"]) + "”", 31)), color=INK, fontsize=5.8,
                    va="top", linespacing=1.28, style="italic")
            xA = px + LABW + 0.04; xB = xA + P + CG
            pt = y - 0.28; pb = pt - P
            for x0, iid, tag, tcol, frame in ((xA, e["gt"], "target", BLUE, True),
                                               (xB, e["other"], "search's first pick" if ok else "verifier's first pick", MUTED, False)):
                ax.imshow(photo(images, files, iid), extent=[x0, x0 + P, pb, pt], zorder=2, interpolation="lanczos")
                ax.add_patch(Rectangle((x0, pb), P, P, fill=False, ec=BLUE if frame else RULE, lw=1.6 if frame else 0.5, zorder=3))
                ax.text(x0, pt + 0.025, tag, color=tcol, fontsize=5.6, fontweight="bold" if frame else "normal", va="bottom")
            ys = pb - 0.075
            ax.text(px, ys, "Search similarity (rank)", color=MUTED, fontsize=FS, va="center")
            for x0, s, rk in ((xA, e["sim"][0], e["srank"][0]), (xB, e["sim"][1], e["srank"][1])):
                ax.text(x0 + P / 2, ys, f"{s:.4f} ({rk})", color=MUTED, fontsize=FS, ha="center", va="center")
            yy = ys - 0.075
            ax.plot([px, xB + P], [yy, yy], color=RULE, lw=0.5)
            for q in e["rows"]:
                lines = label(q); rh = 0.112 + 0.083 * (len(lines) - 1)
                ax.text(px, yy - rh / 2, "\n".join(lines), color=INK, fontsize=FS, va="center", linespacing=1.15)
                for x0, p in ((xA, q["p"][0]), (xB, q["p"][1])):
                    p = 0.0 if p is None else p
                    no = p < 0.5 and q["id"] != "relevance"
                    ax.add_patch(Rectangle((x0 + 0.015, yy - rh + 0.01), P - 0.03, rh - 0.02, fc=RAMP(p), ec=ORANGE if no else "none",
                                           lw=0.8, zorder=3))
                    txt = f"{1 + 4 * p:.1f}" if q["id"] == "relevance" else f"{p:.2f}"
                    ax.text(x0 + P / 2, yy - rh / 2, txt, color="white" if p >= 0.55 else INK, fontsize=5.8, fontweight="bold",
                            ha="center", va="center", zorder=4)
                yy -= rh
            ax.plot([px, xB + P], [yy - 0.005, yy - 0.005], color=RULE, lw=0.5)
            yv = yy - 0.085
            ax.text(px, yv, "Verifier score", color=INK, fontsize=5.9, fontweight="bold", va="center")
            fa, fb = e["final"]
            ax.text(xA + P / 2, yv, f"{fa:.2f}", color=BLUE if fa > fb else INK, fontsize=6.4, fontweight="bold", ha="center", va="center")
            ax.text(xB + P / 2, yv, f"{fb:.2f}", color=ORANGE if fb > fa else INK, fontsize=6.4, fontweight="bold", ha="center", va="center")
        ytop -= rowsH[r] + 0.14
        if r < len(rowsH) - 1:
            ax.plot([0.04, W - 0.04], [ytop + 0.07, ytop + 0.07], color=RULE, lw=0.5)
    fig.savefig(OUT / "example_lookalikes.pdf")
    plt.close(fig)
    print("  wrote figures/example_lookalikes.pdf")


def anole(images: Path, files: dict) -> None:
    t, show = ANOLE
    Q, P = load("docci_all")
    texts = questions("docci_all")[t]
    f = {p.image_id: s for p, s in rank(P[t])}
    rel = relevant(Q[t])
    top3 = sorted(P[t], key=lambda p: (-p.search_score, p.search_rank))[:3]
    query = Q[t]["query"].strip()
    two = ". ".join(query.split(". ")[:2]).rstrip(".") + ". ..."
    plt.rcParams.update({"font.size": 6.6})
    fig = plt.figure(figsize=(3.3, 2.9))
    fig.text(0.02, 0.975, "“" + two + "”", fontsize=5.9, color=MUTED, va="top", wrap=True)
    for k, p in enumerate(top3):
        tgt = p.image_id in rel
        ax = fig.add_axes([0.02 + k * 0.33, 0.43, 0.30, 0.42])
        im = ImageOps.exif_transpose(Image.open(images / files[p.image_id])).convert("RGB"); w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((360, 360))
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BLUE if tgt else "#d8d7d2"); sp.set_linewidth(2.0 if tgt else 0.6)
        ax.set_title("Search #%d%s" % (k + 1, ", the target" if tgt else ""), fontsize=6.2, color=INK, pad=2)
        fig.text(0.02 + k * 0.33 + 0.15, 0.415, f"similarity {p.search_score:.4f}\nverifier score {f[p.image_id]:.2f}", ha="center",
                 va="top", fontsize=6.2, color=BLUE if tgt else INK)
    fig.text(0.02, 0.305, "Probability of yes to three of the five statements:", fontsize=5.6, color=MUTED, va="center")
    y = 0.255
    for qid in show:
        txt = texts[qid]
        fig.text(0.02, y, txt if len(txt) < 66 else txt[:63] + "...", fontsize=5.9, color=INK, va="center")
        for k, p in enumerate(top3):
            v = answer(p, qid) or 0.0
            fig.text(0.02 + k * 0.33 + 0.15, y - 0.04, f"{v:.2f}", fontsize=6.0, va="center", ha="center", color=BLUE if v >= 0.5 else ORANGE)
        y -= 0.085
    fig.savefig(OUT / "example_anole.pdf")
    plt.close(fig)
    print("  wrote figures/example_anole.pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder with DOCCI's test images (test_NNNNN.jpg)")
    a = ap.parse_args()
    files = {r["image_id"]: r["file_name"] for r in pq.read_table(data_root() / "docci/images.parquet").to_pylist()}
    OUT.mkdir(parents=True, exist_ok=True)
    anole(a.images, files)
    lookalikes(a.images, files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
