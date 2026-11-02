# figure_2_coupling_fractions.py

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
    my_style,
    calculate_best_model,
)

from opt import get_data


# =====================================================================
# PATHS
# =====================================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

# Original paper results:
# contains vANN and dANN-LRF
DATA_ORIGINAL = (
    ROOT
    / "DATA"
    / "results_fmnist_1_layer"
)

# ---------------------------------------------------------
# dANN-C folders
# ---------------------------------------------------------
#
# IMPORTANT:
# Check that these directory names exactly match yours.
#

C_DIRS = {

    0.20: (
        ROOT
        / "DATA_C"
        / "results_fmnist_1_layer"
        / "dend_ann_local_rfs_C"
    ),

    0.30: (
        ROOT
        / "DATA_C"
        / "results_fmnist_1_layer"
        / "dend_ann_local_rfs_C_frac30_lam0p1"
    ),

    0.40: (
        ROOT
        / "DATA_C"
        / "results_fmnist_1_layer"
        / "dend_ann_local_rfs_C_frac40_lam0p1"
    ),
}


OUT = (
    ROOT
    / "FinalFigs_manuscript"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SAVE = (
    OUT
    / "figure_2_vANN_dANN_LRF_C10_C20_C40.pdf"
)


# =====================================================================
# SETTINGS
# =====================================================================

SIGMA = 0.0

# Use exactly the same 5 initializations for every model
TRIALS = [1, 2, 3, 4, 5]


# =====================================================================
# STYLE
# =====================================================================

sns.set_style("white")

plt.rcParams.update(
    my_style()
)


# ---------------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------------

palette = {

    # original uncoupled models
    "vANN": "#8d8d8d",
    "dANN-LRF": "#ff9f9b",

    # coupled dANNs
    "dANN-C-20": "#fcae91",
    "dANN-C-30": "#cb181d",
    "dANN-C-40": "#67000d",
}


order = [
    "vANN",
    "dANN-LRF",
    "dANN-C-20",
    "dANN-C-30",
    "dANN-C-40",
]


# =====================================================================
# NUMBER OF TRAINABLE PARAMETERS
# =====================================================================

def nparams(
    D,
    S,
    n_syn=16,
    n_class=10,
):
    """
    Number of trainable parameters for the one-layer dANN architecture.

    Coupling itself adds no trainable parameters.
    """

    return (
        (n_syn + 1) * D * S
        + (D + 1) * S
        + (S + 1) * n_class
    )


# =====================================================================
# ORIGINAL PAPER DATA
# =====================================================================

print("\nLoading original vANN and dANN-LRF data...")

with open(
    DATA_ORIGINAL / "output_all_final.pkl",
    "rb",
) as f:

    original_results = pickle.load(f)


df_original = fix_names(
    original_results["testing"]
).copy()


# Original data stores accuracy as 0–1
df_original["test_acc"] *= 100


# Keep only the two models we want
df_original = df_original[
    df_original["model"].isin(
        [
            "vANN",
            "dANN-LRF",
        ]
    )
].copy()


# Keep trials 1–5 explicitly
df_original = df_original[
    df_original["trial"].isin(TRIALS)
].copy()


print("\nOriginal models:")

print(
    df_original.groupby("model")
    .size()
)


# =====================================================================
# LOAD dANN-C RESULTS
# =====================================================================

pattern = re.compile(
    r"results_sigma_0\.0_"
    r"trial_(\d+)_"
    r"dends_(\d+)_"
    r"soma_(\d+)\.pkl"
)


def load_coupled_results(
    folder,
    model_name,
):
    """
    Load all D x S architectures for one coupling condition.

    Only trials 1–5 are retained.
    """

    folder = pathlib.Path(folder)

    if not folder.exists():

        raise FileNotFoundError(
            f"\nFolder does not exist:\n"
            f"{folder}\n"
        )

    rows = []

    files = folder.glob(
        "results_sigma_0.0_trial_*_dends_*_soma_*.pkl"
    )

    for file in files:

        m = pattern.fullmatch(
            file.name
        )

        if m is None:
            continue

        trial, D, S = map(
            int,
            m.groups(),
        )

        # -------------------------------------------------
        # Balanced comparison:
        # trials 1–5 only
        # -------------------------------------------------

        if trial not in TRIALS:
            continue

        with open(
            file,
            "rb",
        ) as f:

            out = pickle.load(f)

        rows.append(
            {
                "model": model_name,

                "trial": trial,

                "num_dends": D,

                "num_soma": S,

                "test_loss": float(
                    np.asarray(
                        out["test_loss"]
                    ).squeeze()
                ),

                "test_acc": (
                    float(
                        np.asarray(
                            out["test_acc"]
                        ).squeeze()
                    )
                    * 100
                ),

                "trainable_params": nparams(
                    D,
                    S,
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    print(
        f"\n{model_name}"
    )

    print(
        f"Folder:\n{folder}"
    )

    print(
        f"Runs loaded: {len(df)}"
    )

    if not df.empty:

        print(
            "Architectures:",
            df[
                [
                    "num_dends",
                    "num_soma",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        print(
            "Trials:",
            sorted(
                df["trial"].unique()
            )
        )

    return df


# =====================================================================
# LOAD C20 / C30 / C40
# =====================================================================

df_C20 = load_coupled_results(
    C_DIRS[0.20],
    "dANN-C-20",
)

df_C30 = load_coupled_results(
    C_DIRS[0.30],
    "dANN-C-30",
)

df_C40 = load_coupled_results(
    C_DIRS[0.40],
    "dANN-C-40",
)


# =====================================================================
# COMBINE
# =====================================================================

df = pd.concat(
    [
        df_original,
        df_C20,
        df_C30,
        df_C40,
    ],
    ignore_index=True,
)


# Make model ordering explicit
df["model"] = pd.Categorical(
    df["model"],
    categories=order,
    ordered=True,
)


print("\n====================================")
print("FINAL DATASET")
print("====================================")

print(
    df.groupby(
        "model",
        observed=True,
    ).size()
)


# =====================================================================
# CHECK EACH COUPLED CONDITION
# =====================================================================

print("\nArchitecture counts:")

for model_name in [
    "dANN-C-20",
    "dANN-C-30",    
    "dANN-C-40",
]:

    tmp = df[
        df["model"] == model_name
    ]

    counts = (
        tmp.groupby(
            [
                "num_dends",
                "num_soma",
            ],
            observed=True,
        )
        .size()
    )

    print(
        f"\n{model_name}:"
    )

    print(
        counts.value_counts()
    )


# =====================================================================
# vANN REFERENCE
# =====================================================================

df_vann = df_original[
    df_original["model"] == "vANN"
].copy()


eval_metrics = calculate_best_model(
    df_vann
)


# Lowest vANN loss
v_loss = np.nanmin(
    eval_metrics[:, 0]
)

v_loss_params = (
    eval_metrics[
        np.nanargmin(
            eval_metrics[:, 0]
        )
    ][-1]
)


# Highest vANN accuracy
v_acc = np.nanmax(
    eval_metrics[:, 1]
)

v_acc_params = (
    eval_metrics[
        np.nanargmax(
            eval_metrics[:, 1]
        )
    ][-1]
)


print(
    f"\nBest vANN loss: "
    f"{v_loss:.4f}"
)

print(
    f"Best vANN accuracy: "
    f"{v_acc:.3f}%"
)


# =====================================================================
# REPORT BEST ARCHITECTURE FOR EACH MODEL
# =====================================================================

print("\n====================================")
print("BEST ARCHITECTURES")
print("====================================")


for model_name in order:

    dm = df[
        df["model"] == model_name
    ].copy()

    if dm.empty:
        continue

    means = (
        dm.groupby(
            [
                "num_dends",
                "num_soma",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            test_acc=(
                "test_acc",
                "mean",
            ),

            test_loss=(
                "test_loss",
                "mean",
            ),
        )
    )

    best = means.loc[
        means["test_acc"].idxmax()
    ]

    print(
        f"\n{model_name}:"
        f"\n  D = {int(best.num_dends)}"
        f"\n  S = {int(best.num_soma)}"
        f"\n  accuracy = {best.test_acc:.3f}%"
        f"\n  loss = {best.test_loss:.4f}"
    )


# =====================================================================
# FIGURE
# =====================================================================

fig = plt.figure(
    figsize=(
        7.086614 * 0.98,
        11.69 * 0.70,
    ),
    layout="constrained",
)


# Panel A on top;
# B-E underneath
top_fig, bottom_fig = fig.subfigures(
    2,
    1,
    height_ratios=[
        1,
        3.5,
    ],
)


# =====================================================================
# PANEL A — FMNIST examples
# =====================================================================

mosaic_images = [

    [
        "a",
        "b",
        "c",
        "d",
        "e",
    ],

    [
        "f",
        "g",
        "h",
        "i",
        "j",
    ],
]


ax_img = top_fig.subplot_mosaic(
    mosaic_images,
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
    0.1,
    "fmnist",
    seed=0,
)


x_train = data["train"]
y_train = labels["train"]


for i, ax in enumerate(
    ax_img.values()
):

    x = (
        x_train[
            y_train == i
        ][0]
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

    ax.axis(
        "off"
    )


# =====================================================================
# PANELS B-E
# =====================================================================

axd = bottom_fig.subplot_mosaic(
    [
        [
            "b",
            "c",
        ],

        [
            "d",
            "e",
        ],
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


# =====================================================================
# PANEL B
# Test loss vs trainable parameters
# =====================================================================

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

    errorbar="se",

    legend=False,

    ax=axd["b"],
)


axd["b"].axhline(
    v_loss,
    ls="--",
    c="k",
    lw=1,
)


axd["b"].axvline(
    v_loss_params,
    ls="--",
    c="k",
    lw=1,
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


# =====================================================================
# PANEL C
# Test accuracy vs trainable parameters
# =====================================================================

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

    errorbar="se",

    ax=axd["c"],
)


axd["c"].axhline(
    v_acc,
    ls="--",
    c="k",
    lw=1,
)


axd["c"].axvline(
    v_acc_params,
    ls="--",
    c="k",
    lw=1,
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


# =====================================================================
# PANELS D/E
#
# Same basic Figure-2 logic:
# focus on the large soma populations S = 256 / 512.
#
# vANN has no meaningful dendrites-per-soma axis,
# so it is represented by its horizontal reference.
# =====================================================================

df_ds = df[
    (
        df["num_soma"] > 200
    )
    &
    (
        df["model"] != "vANN"
    )
].copy()


# =====================================================================
# PANEL D
# Test loss vs dendrites / soma
# =====================================================================

sns.lineplot(
    data=df_ds,

    x="num_dends",
    y="test_loss",

    hue="model",
    style="num_soma",

    hue_order=[
        "dANN-LRF",
        "dANN-C-20",
        "dANN-C-30",
        "dANN-C-40",
    ],

    markers=True,

    palette=palette,

    errorbar="se",

    legend=False,

    ax=axd["d"],
)


axd["d"].axhline(
    v_loss,
    ls="--",
    c="k",
    lw=1,
)


axd["d"].set_xscale(
    "log",
    base=2,
)


axd["d"].set_xticks(
    [
        2,
        4,
        8,
        16,
        32,
        64,
    ]
)


axd["d"].set_xticklabels(
    [
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
    ]
)


axd["d"].set_xlabel(
    "dendrites per soma"
)


axd["d"].set_ylabel(
    "test loss"
)


# =====================================================================
# PANEL E
# Test accuracy vs dendrites / soma
# =====================================================================

sns.lineplot(
    data=df_ds,

    x="num_dends",
    y="test_acc",

    hue="model",
    style="num_soma",

    hue_order=[
        "dANN-LRF",
        "dANN-C-20",
        "dANN-C-30",
        "dANN-C-40",
    ],

    markers=True,

    palette=palette,

    errorbar="se",

    ax=axd["e"],
)


axd["e"].axhline(
    v_acc,
    ls="--",
    c="k",
    lw=1,
)


axd["e"].set_xscale(
    "log",
    base=2,
)


axd["e"].set_xticks(
    [
        2,
        4,
        8,
        16,
        32,
        64,
    ]
)


axd["e"].set_xticklabels(
    [
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
    ]
)


axd["e"].set_xlabel(
    "dendrites per soma"
)


axd["e"].set_ylabel(
    "test accuracy (%)"
)


# =====================================================================
# LEGEND
# =====================================================================

handles, labels_legend = (
    axd["c"]
    .get_legend_handles_labels()
)


# remove default legend from C
if axd["c"].legend_ is not None:
    axd["c"].legend_.remove()


# E also has a style legend;
# remove it because we make one main model legend
if axd["e"].legend_ is not None:
    axd["e"].legend_.remove()


# Only retain our five model labels
legend_dict = {}

for hnd, lab in zip(
    handles,
    labels_legend,
):

    if lab in order:

        legend_dict[lab] = hnd


fig.legend(
    [
        legend_dict[m]
        for m in order
        if m in legend_dict
    ],

    [
        m
        for m in order
        if m in legend_dict
    ],

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        1.01,
    ),

    ncol=5,

    frameon=False,
)


# =====================================================================
# FINAL STYLE
# =====================================================================

for ax in axd.values():

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


# =====================================================================
# SAVE
# =====================================================================

fig.savefig(
    SAVE,
    dpi=600,
    bbox_inches="tight",
    transparent=True,
)


fig.savefig(
    SAVE.with_suffix(".svg"),
    bbox_inches="tight",
    transparent=True,
)


print(
    f"\nSaved PDF:\n{SAVE}"
)

print(
    f"\nSaved SVG:\n"
    f"{SAVE.with_suffix('.svg')}"
)