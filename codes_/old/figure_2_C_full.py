# figure_2_C_full.py

import os
import re
import pickle
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import (
    fix_names,
    keep_models,
    my_style,
    calculate_best_model,
)
from opt import get_data


# ============================================================
# PATHS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

DATA = ROOT / "DATA/results_fmnist_1_layer"

DATA_C = (
    ROOT
    / "DATA_C/results_fmnist_1_layer/dend_ann_local_rfs_C"
)

OUT = ROOT / "FinalFigs_manuscript"
OUT.mkdir(exist_ok=True)


# ============================================================
# PLOT STYLE
# ============================================================

sns.set_style("white")
plt.rcParams.update(my_style())

palette = {
    "dANN-R": "#8de5a1",
    "dANN-LRF": "#ff9f9b",
    "dANN-GRF": "#a1c9f4",
    "pdANN": "#d0bbff",
    "vANN": "#8d8d8d",
    "dANN-C": "#7f0000",
}


# ============================================================
# ORIGINAL PAPER DATA
# ============================================================

with open(DATA / "output_all_final.pkl", "rb") as f:
    results = pickle.load(f)

df_test = fix_names(results["testing"]).copy()

# Convert test accuracy from fraction to %
df_test["test_acc"] *= 100

models_original = [
    "dANN-R",
    "dANN-LRF",
    "dANN-GRF",
    "pdANN",
    "vANN",
]

df_old = keep_models(
    df_test,
    models_original,
).copy()


# ============================================================
# dANN-C DATA
# ============================================================

def nparams(D, S, n_syn=16, n_class=10):
    """
    Number of trainable parameters for the structured dANN architecture.

    Coupling adds no trainable parameters because lambda is fixed.
    """
    return (
        (n_syn + 1) * D * S
        + (D + 1) * S
        + (S + 1) * n_class
    )


rows = []

pat = re.compile(
    r"results_sigma_0\.0_trial_(\d+)_dends_(\d+)_soma_(\d+)\.pkl"
)

for p in DATA_C.glob(
    "results_sigma_0.0_trial_*_dends_*_soma_*.pkl"
):

    m = pat.fullmatch(p.name)

    if not m:
        continue

    trial, D, S = map(int, m.groups())

    with open(p, "rb") as f:
        out = pickle.load(f)

    rows.append({
        "model": "dANN-C",
        "trial": trial,
        "num_dends": D,
        "num_soma": S,
        "test_loss": float(
            np.asarray(out["test_loss"]).squeeze()
        ),
        "test_acc": float(
            np.asarray(out["test_acc"]).squeeze()
        ) * 100,
        "trainable_params": nparams(D, S),
    })


df_C = pd.DataFrame(rows)


# ============================================================
# CHECK RAW dANN-C RUNS
# ============================================================

print("\n==========================================")
print("dANN-C DATA CHECK")
print("==========================================")

print(f"All dANN-C runs found: {len(df_C)}")

counts_before = (
    df_C
    .groupby(["num_dends", "num_soma"])
    .size()
)

print("\nRuns per D × S before filtering:")
print(
    counts_before
    .value_counts()
    .sort_index()
)


# ============================================================
# FIGURE 2 USES ONLY TRIALS 1-5
# ============================================================
#
# The original Figure 2 uses N = 5 independent initializations
# per architecture.
#
# We subsequently generated trials 6-15 specifically for
# D = 8, S = 512.
#
# Those extra trials are useful for dedicated dANN-C analyses,
# but MUST NOT be included in the Figure 2 architecture grid.
# ============================================================

df_C = df_C[
    df_C["trial"].between(1, 5)
].copy()


print(
    f"\ndANN-C runs used for Figure 2: "
    f"{len(df_C)}"
)

counts = (
    df_C
    .groupby(["num_dends", "num_soma"])
    .size()
)

print("\nRuns per D × S after filtering:")
print(
    counts
    .value_counts()
    .sort_index()
)


# ============================================================
# SANITY CHECKS
# ============================================================

dends_order = [2, 4, 8, 16, 32, 64]
soma_order = [32, 64, 128, 256, 512]

expected_architectures = (
    len(dends_order) * len(soma_order)
)

expected_trials = 5

expected_runs = (
    expected_architectures
    * expected_trials
)


assert len(counts) == expected_architectures, (
    f"Expected {expected_architectures} D×S "
    f"architectures, found {len(counts)}"
)

assert (counts == expected_trials).all(), (
    "Some D×S architectures do not contain "
    "exactly 5 trials:\n"
    f"{counts[counts != expected_trials]}"
)

assert len(df_C) == expected_runs, (
    f"Expected {expected_runs} dANN-C runs "
    f"after filtering, found {len(df_C)}"
)


# ============================================================
# COMBINE ORIGINAL DATA + dANN-C
# ============================================================

df = pd.concat(
    [
        df_old,
        df_C,
    ],
    ignore_index=True,
    sort=False,
)


# Plot order.
# dANN-C is placed next to dANN-LRF because it is the
# corresponding coupled version of that architecture.
order = [
    "dANN-R",
    "dANN-LRF",
    "dANN-C",
    "dANN-GRF",
    "pdANN",
    "vANN",
]


print("\n==========================================")
print("COMBINED FIGURE 2 DATA")
print("==========================================")

print("\nModels included:")
print(
    df["model"]
    .value_counts()
)

print(
    f"\nCombined dataframe shape: "
    f"{df.shape}"
)


# ============================================================
# CHECK dANN-C PERFORMANCE
# ============================================================

# Best vANN accuracy from the original paper data.
# Mean across trials is calculated for each architecture first.

vann = df_old[
    df_old["model"] == "vANN"
].copy()

vann_mean = (
    vann
    .groupby(
        ["num_dends", "num_soma"]
    )["test_acc"]
    .mean()
)

vann_best_acc = vann_mean.max()

print("\n==========================================")
print("PERFORMANCE CHECK")
print("==========================================")

print(
    f"Best vANN accuracy: "
    f"{vann_best_acc:.3f}%"
)


# Mean dANN-C performance across trials 1-5
C_mean = (
    df_C
    .groupby(
        ["num_dends", "num_soma"],
        as_index=False,
    )
    .agg(
        test_acc=("test_acc", "mean"),
        test_loss=("test_loss", "mean"),
    )
)


# ------------------------------------------------------------
# Select the first dANN-C architecture that exceeds the
# best vANN accuracy, following Figure 2 architecture ordering
# ------------------------------------------------------------

selected = None

for D in dends_order:

    for S in soma_order:

        row = C_mean[
            (C_mean["num_dends"] == D)
            &
            (C_mean["num_soma"] == S)
        ]

        if (
            len(row)
            and
            row.iloc[0]["test_acc"]
            > vann_best_acc
        ):

            selected = row.iloc[0]
            break

    if selected is not None:
        break


# Fallback:
# if no architecture beats vANN,
# choose the highest-accuracy dANN-C architecture.
if selected is None:

    selected = C_mean.loc[
        C_mean["test_acc"].idxmax()
    ]


print("\nSelected dANN-C:")
print(
    f"D = {int(selected.num_dends)}, "
    f"S = {int(selected.num_soma)}, "
    f"accuracy = {selected.test_acc:.3f}%, "
    f"loss = {selected.test_loss:.4f}"
)


# ------------------------------------------------------------
# Absolute highest-accuracy architecture
# ------------------------------------------------------------

top = C_mean.loc[
    C_mean["test_acc"].idxmax()
]

print("\nAbsolute highest-accuracy dANN-C:")
print(
    f"D = {int(top.num_dends)}, "
    f"S = {int(top.num_soma)}, "
    f"accuracy = {top.test_acc:.3f}%, "
    f"loss = {top.test_loss:.4f}"
)


# ============================================================
# VANILLA REFERENCE LINES
# ============================================================

eval_metrics = calculate_best_model(
    df_old[
        df_old["model"] == "vANN"
    ]
)


# Best loss benchmark
v_loss = np.nanmin(
    eval_metrics[:, 0]
)

v_loss_params = eval_metrics[
    np.nanargmin(
        eval_metrics[:, 0]
    )
][-1]


# Best accuracy benchmark
v_acc = np.nanmax(
    eval_metrics[:, 1]
)

v_acc_params = eval_metrics[
    np.nanargmax(
        eval_metrics[:, 1]
    )
][-1]


print("\nVanilla reference:")
print(
    f"minimum loss = {v_loss:.4f} "
    f"at {int(v_loss_params)} parameters"
)

print(
    f"maximum accuracy = {v_acc:.3f}% "
    f"at {int(v_acc_params)} parameters"
)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        7.086614 * 0.98,
        11.69 * 0.7,
    ),
    layout="constrained",
)


top_fig, bottom_fig = fig.subfigures(
    nrows=2,
    ncols=1,
    height_ratios=[
        1,
        3.5,
    ],
)


# ============================================================
# PANEL A
# FMNIST EXAMPLES
# ============================================================

mosaic_l = [
    ["a", "b", "c", "d", "e"],
    ["f", "g", "h", "i", "j"],
]

ax_img = top_fig.subplot_mosaic(
    mosaic_l,
    gridspec_kw={
        "wspace": 0,
        "hspace": 0,
        "left": 0,
    },
)


trans = mtransforms.ScaledTranslation(
    -20 / 72,
    7 / 72,
    fig.dpi_scale_trans,
)


ax_img["a"].text(
    0,
    1,
    "a",
    transform=(
        ax_img["a"].transAxes
        + trans
    ),
    fontsize="large",
    va="bottom",
)


data, labels, h, w, ch = get_data(
    validation_split=0.1,
    dtype="fmnist",
    seed=0,
)

x_train = data["train"]
y_train = labels["train"]


for i, ax in enumerate(
    ax_img.values()
):

    x = (
        x_train[y_train == i][0]
        .reshape(
            w,
            h,
            ch,
        )
        .squeeze()
    )

    ax.imshow(
        x,
        cmap="gray",
    )

    ax.axis("off")


# ============================================================
# PANELS B-E
# ============================================================

axd = bottom_fig.subplot_mosaic(
    [
        ["b", "c"],
        ["d", "e"],
    ]
)


for label, ax in axd.items():

    ax.text(
        0,
        1,
        label,
        transform=(
            ax.transAxes
            + trans
        ),
        fontsize="large",
        va="bottom",
    )


# ============================================================
# PANEL B
# TEST LOSS vs TRAINABLE PARAMETERS
# ============================================================

sns.lineplot(
    data=df,
    x="trainable_params",
    y="test_loss",
    hue="model",
    style="model",
    hue_order=order,
    style_order=order,
    markers=True,
    dashes=False,
    palette=palette,
    errorbar=("ci", 95),
    legend=False,
    ax=axd["b"],
)


axd["b"].axhline(
    v_loss,
    linestyle="--",
    color="k",
    linewidth=1.5,
)

axd["b"].axvline(
    v_loss_params,
    linestyle="--",
    color="k",
    linewidth=1.5,
)

axd["b"].set_xscale(
    "log"
)

axd["b"].set_xlabel(
    "trainable params"
)

axd["b"].set_ylabel(
    "test loss"
)


# ============================================================
# PANEL C
# TEST ACCURACY vs TRAINABLE PARAMETERS
# ============================================================

sns.lineplot(
    data=df,
    x="trainable_params",
    y="test_acc",
    hue="model",
    style="model",
    hue_order=order,
    style_order=order,
    markers=True,
    dashes=False,
    palette=palette,
    errorbar=("ci", 95),
    ax=axd["c"],
)


axd["c"].axhline(
    v_acc,
    linestyle="--",
    color="k",
    linewidth=1.5,
)

axd["c"].axvline(
    v_acc_params,
    linestyle="--",
    color="k",
    linewidth=1.5,
)

axd["c"].set_xscale(
    "log"
)

axd["c"].set_xlabel(
    "trainable params"
)

axd["c"].set_ylabel(
    "test accuracy (%)"
)


# ============================================================
# PANELS D-E
# SAME SOMA RESTRICTION AS ORIGINAL FIGURE
# ============================================================

df_de = df[
    (df["num_soma"] > 200)
    &
    (df["model"] != "vANN")
].copy()


# Explicit model ordering for panels D/E
order_de = [
    model
    for model in order
    if model != "vANN"
]


# ============================================================
# PANEL D
# TEST LOSS vs DENDRITES PER SOMA
# ============================================================

sns.lineplot(
    data=df_de,
    x="num_dends",
    y="test_loss",
    hue="model",
    style="num_soma",
    hue_order=order_de,
    markers=True,
    markersize=7,
    palette=palette,
    errorbar=("ci", 95),
    legend=False,
    ax=axd["d"],
)


axd["d"].axhline(
    v_loss,
    linestyle="--",
    color="k",
    linewidth=1.5,
)

axd["d"].set_xscale(
    "log",
    base=2,
)

axd["d"].set_xlabel(
    "dendrites per soma"
)

axd["d"].set_ylabel(
    "test loss"
)


# ============================================================
# PANEL E
# TEST ACCURACY vs DENDRITES PER SOMA
# ============================================================

sns.lineplot(
    data=df_de,
    x="num_dends",
    y="test_acc",
    hue="model",
    style="num_soma",
    hue_order=order_de,
    markers=True,
    markersize=7,
    palette=palette,
    errorbar=("ci", 95),
    ax=axd["e"],
)


axd["e"].axhline(
    v_acc,
    linestyle="--",
    color="k",
    linewidth=1.5,
)

axd["e"].set_xscale(
    "log",
    base=2,
)

axd["e"].set_xlabel(
    "dendrites per soma"
)

axd["e"].set_ylabel(
    "test accuracy (%)"
)


# ============================================================
# SAVE
# ============================================================

svg_path = (
    OUT
    / "figure_2_with_dANNC.svg"
)

pdf_path = (
    OUT
    / "figure_2_with_dANNC.pdf"
)


fig.savefig(
    svg_path,
    bbox_inches="tight",
    dpi=600,
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)


plt.close(fig)


print("\n==========================================")
print("FIGURE SAVED")
print("==========================================")

print(svg_path)
print(pdf_path)