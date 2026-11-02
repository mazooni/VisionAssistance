#!/usr/bin/env python3
"""Render the benchmark charts for the VisionAssistance README.

Usage: python3 make_charts.py <benchmarks_v1 dir> <output dir>

Writes <name>_light.png and <name>_dark.png for each chart, with a
transparent background so GitHub's <picture> element can pick the variant.
Numbers come straight from benchmarks_v1/results/*.json and
benchmarks_v1/results_strat/*.json.
"""
import json
import pathlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

warnings.filterwarnings("ignore")

RES = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

DPI = 200
S = DPI / 100.0  # file pixels per "display" pixel (README shows at half size)
HAIR = 2 / (DPI / 72)  # 2 file px -> points (1 display px hairline)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.unicode_minus": False,
})

MODELS = [
    ("ours_v3", "VisionAssist YOLO26n v3 (ours)"),
    ("yolo26n", "YOLO26n, COCO weights"),
    ("yolo12n", "YOLOv12n, COCO weights"),
]
BENCH = [
    ("cocoval", "COCO val2017"),
    ("wotr", "WOTR"),
    ("mapillary", "Mapillary Vistas"),
]
BINS = [
    ("tiny", "tiny\n< 0.1 %"),
    ("small", "small\n0.1–1 %"),
    ("medium", "medium\n1–5 %"),
    ("large", "large\n> 5 %"),
]

# Reference palette (dataviz skill): categorical slots 1-3, stepped per mode.
THEMES = {
    "light": dict(series=["#2a78d6", "#eb6834", "#1baf7a"], ink="#0b0b0b",
                  ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  base="#c3c2b7", surface="#ffffff"),
    "dark": dict(series=["#3987e5", "#d95926", "#199e70"], ink="#f0efec",
                 ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 base="#383835", surface="#0d1117"),
}


def load(kind, bench, model):
    with open(RES / kind / f"{bench}__{model}.json") as f:
        return json.load(f)


def pretty(cls):
    return cls.replace("_", " ")


# --- rounded-end bars -------------------------------------------------------
K = 0.5522847  # cubic Bezier circle constant


def bar_path(x0, y0, length, thick, rx, ry, horizontal):
    """Bar anchored at (x0, y0) with a 4px-rounded data end, square baseline."""
    if horizontal:
        rx, ry = min(rx, length), min(ry, thick / 2)
        L, h = length, thick
        v = [(x0, y0), (x0 + L - rx, y0),
             (x0 + L - rx + K * rx, y0), (x0 + L, y0 + ry - K * ry), (x0 + L, y0 + ry),
             (x0 + L, y0 + h - ry),
             (x0 + L, y0 + h - ry + K * ry), (x0 + L - rx + K * rx, y0 + h), (x0 + L - rx, y0 + h),
             (x0, y0 + h), (x0, y0)]
        c = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CLOSEPOLY]
    else:
        rx, ry = min(rx, thick / 2), min(ry, length)
        L, h = length, thick
        v = [(x0, y0), (x0 + h, y0), (x0 + h, y0 + L - ry),
             (x0 + h, y0 + L - ry + K * ry), (x0 + h - rx + K * rx, y0 + L), (x0 + h - rx, y0 + L),
             (x0 + rx, y0 + L),
             (x0 + rx - K * rx, y0 + L), (x0, y0 + L - ry + K * ry), (x0, y0 + L - ry),
             (x0, y0)]
        c = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    return Path(v, c)


def px2data(ax, px):
    """Convert a length in file pixels to (dx, dy) data units."""
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    a = inv.transform((0, 0))
    b = inv.transform((px, px))
    return abs(b[0] - a[0]), abs(b[1] - a[1])


def bar(ax, x0, y0, length, thick, color, horizontal):
    rx, ry = px2data(ax, 4 * S)
    ax.add_patch(PathPatch(bar_path(x0, y0, length, thick, rx, ry, horizontal),
                           facecolor=color, edgecolor="none", zorder=3))


# --- chrome -----------------------------------------------------------------
def style(ax, t, value_axis):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, labelsize=9, colors=t["ink2"])
    ax.set_axisbelow(True)
    if value_axis == "y":
        ax.yaxis.grid(True, color=t["grid"], linewidth=HAIR)
        ax.axhline(0, color=t["base"], linewidth=HAIR, zorder=2)
    else:
        ax.xaxis.grid(True, color=t["grid"], linewidth=HAIR)
        ax.axvline(0, color=t["base"], linewidth=HAIR, zorder=2)


def heading(fig, t, title, subtitle, y=0.955):
    fig.text(0.04, y, title, fontsize=13, fontweight="semibold", color=t["ink"],
             ha="left", va="top")
    dy = 0.27 / fig.get_figheight()  # fixed 0.27 in below the title
    fig.text(0.04, y - dy, subtitle, fontsize=9, color=t["muted"], ha="left",
             va="top")


def legend(fig, t, y, labels=None):
    labels = labels or [ml for _, ml in MODELS]
    handles = [Patch(facecolor=c, edgecolor="none") for c in t["series"][:len(labels)]]
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.035, y),
                     ncol=len(labels), frameon=False, fontsize=9,
                     handlelength=1.0, handleheight=1.0, columnspacing=1.6,
                     handletextpad=0.6)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])


def save(fig, name, theme):
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=DPI, transparent=True)
    plt.close(fig)


# --- chart 1: shared-class mAP50 per benchmark --------------------------------
def chart_shared(theme, t):
    fig = plt.figure(figsize=(8, 4.6), dpi=DPI)
    heading(fig, t, "Shared-class mAP50 on three public benchmarks",
            "Mean AP50 over the classes all three models can predict, 640 px input. "
            "COCO 16 classes · WOTR 9 · Mapillary 10")
    legend(fig, t, 0.80)
    ax = fig.add_axes([0.075, 0.11, 0.905, 0.60])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 0.65)
    style(ax, t, "y")
    tx, _ = px2data(ax, 22 * S)
    gx, _ = px2data(ax, 2 * S)
    total = 3 * tx + 2 * gx
    for bi, (b, _) in enumerate(BENCH):
        for mi, (m, _) in enumerate(MODELS):
            v = load("results", b, m)["shared"]["mAP50"]
            x0 = bi - total / 2 + mi * (tx + gx)
            bar(ax, x0, 0, v, tx, t["series"][mi], horizontal=False)
            if mi == 0:
                ax.text(x0 + tx / 2, v + 0.012, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=9, color=t["ink"])
    ax.set_xticks(range(3))
    ax.set_xticklabels([bl for _, bl in BENCH], color=t["ink"], fontsize=10)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6"])
    save(fig, "shared_map50", theme)


# --- chart 2: WOTR per-class AP50, nine shared classes ------------------------
def chart_wotr_classes(theme, t):
    ours = load("results", "wotr", "ours_v3")["shared"]["per_class"]
    others = {m: load("results", "wotr", m)["shared"]["per_class"] for m, _ in MODELS[1:]}
    classes = sorted(ours, key=lambda c: -ours[c]["AP50"])
    n = len(classes)

    fig = plt.figure(figsize=(8, 6.4), dpi=DPI)
    heading(fig, t, "WOTR: AP50 on the nine classes all three models share",
            "1,594 street-level images annotated for visually impaired navigation. "
            "Sorted by our model's score.")
    legend(fig, t, 0.855)
    ax = fig.add_axes([0.17, 0.05, 0.80, 0.70])
    ax.set_xlim(0, 0.85)
    ax.set_ylim(n - 0.5, -0.5)  # first class on top
    style(ax, t, "x")
    _, ty = px2data(ax, 13 * S)
    _, gy = px2data(ax, 2 * S)
    total = 3 * ty + 2 * gy
    for ci, cls in enumerate(classes):
        for mi, (m, _) in enumerate(MODELS):
            v = ours[cls]["AP50"] if mi == 0 else others[m][cls]["AP50"]
            y0 = ci - total / 2 + mi * (ty + gy)
            bar(ax, 0, y0, v, ty, t["series"][mi], horizontal=True)
            if mi == 0:
                ax.text(v + 0.008, y0 + ty / 2, f"{v:.3f}", ha="left",
                        va="center", fontsize=8.5, color=t["ink"])
    ax.set_yticks(range(n))
    ax.set_yticklabels([pretty(c) for c in classes], color=t["ink"], fontsize=10)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8"])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", pad=2)
    save(fig, "wotr_shared_classes", theme)


# --- chart 3: custom classes (ours only) --------------------------------------
def chart_custom(theme, t):
    groups = []
    d = load("results", "door", "ours_v3")["custom"]["per_class"]
    s = load("results", "stairs", "ours_v3")["custom"]["per_class"]
    groups.append(("Our held-out test splits",
                   [("door", d["door"]["AP50"]), ("stairs", s["stairs"]["AP50"])]))
    for b, bl in (("wotr", "WOTR"), ("mapillary", "Mapillary Vistas")):
        pc = load("results", b, "ours_v3")["custom"]["per_class"]
        rows = sorted(((c, pc[c]["AP50"]) for c in pc), key=lambda r: -r[1])
        groups.append((bl, rows))

    # row layout: header row, then one row per class, 0.45 of a row between groups
    rows = []  # (y, kind, label, value)
    y = 0.0
    for gi, (gl, items) in enumerate(groups):
        if gi:
            y += 0.45
        rows.append((y, "head", gl, None))
        y += 1
        for cls, v in items:
            rows.append((y, "bar", pretty(cls), v))
            y += 1
    n_rows = y

    fig = plt.figure(figsize=(8, 6.6), dpi=DPI)
    heading(fig, t, "Custom classes: AP50 of our model",
            "Nine classes the COCO-pretrained baselines cannot detect at all, "
            "so there is no baseline bar to compare against.")
    ax = fig.add_axes([0.26, 0.05, 0.71, 0.76])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(n_rows - 0.5, -0.5)
    style(ax, t, "x")
    _, ty = px2data(ax, 16 * S)
    for yy, kind, label, v in rows:
        if kind == "head":
            ax.text(-0.012, yy, label, ha="right", va="center", fontsize=9,
                    color=t["muted"], fontweight="semibold", transform=ax.transData,
                    clip_on=False)
            continue
        bar(ax, 0, yy - ty / 2, v, ty, t["series"][0], horizontal=True)
        ax.text(-0.012, yy, label, ha="right", va="center", fontsize=10,
                color=t["ink"], clip_on=False)
        ax.text(v + 0.008, yy, f"{v:.3f}", ha="left", va="center", fontsize=9,
                color=t["ink"])
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", pad=2)
    save(fig, "custom_classes", theme)


# --- chart 4: size-stratified AP50 --------------------------------------------
def chart_size(theme, t):
    fig = plt.figure(figsize=(10, 4.4), dpi=DPI)
    heading(fig, t, "Shared-class AP50 by relative object size",
            "Ground-truth boxes binned by share of image area. Objects a pedestrian must "
            "react to are a few metres away and land in the medium and large bins.")
    legend(fig, t, 0.80)
    left, width, gap = 0.06, 0.285, 0.03
    for bi, (b, bl) in enumerate(BENCH):
        ax = fig.add_axes([left + bi * (width + gap), 0.13, width, 0.50])
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(0, 1.0)
        style(ax, t, "y")
        tx, _ = px2data(ax, 14 * S)
        gx, _ = px2data(ax, 2 * S)
        total = 3 * tx + 2 * gx
        counts = None
        for mi, (m, _) in enumerate(MODELS):
            data = load("results_strat", b, m)
            counts = counts or data["gt_bin_counts"]
            blk = data["blocks"]["shared"]
            for xi, (key, _) in enumerate(BINS):
                v = blk[key]["mAP50"]
                x0 = xi - total / 2 + mi * (tx + gx)
                bar(ax, x0, 0, v, tx, t["series"][mi], horizontal=False)
        ax.set_xticks(range(4))
        ax.set_xticklabels([lbl for _, lbl in BINS], color=t["ink2"], fontsize=8.5)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if bi == 0:
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        else:
            ax.set_yticklabels([])
        ax.text(0.0, 1.04, bl, transform=ax.transAxes, fontsize=10.5,
                color=t["ink"], fontweight="semibold", ha="left", va="bottom")
        ax.text(1.0, 1.04, f"{counts['all']:,} boxes", transform=ax.transAxes,
                fontsize=8.5, color=t["muted"], ha="right", va="bottom")
    save(fig, "size_bins", theme)


for theme, t in THEMES.items():
    chart_shared(theme, t)
    chart_wotr_classes(theme, t)
    chart_custom(theme, t)
    chart_size(theme, t)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
