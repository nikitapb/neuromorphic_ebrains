# figure_2_models.py

import re
import pickle
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_functions import fix_names, my_style


# ============================================================
# SETTINGS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

# Original Chavlis/Poirazi results:
ORIGINAL_FILE = (
    ROOT
    / "DATA"
    / "results_fmnist_1_layer"
    / "output_all_final.pkl"
)

# dANN-C results:
C_ROOT = (
    ROOT
    / "DATA_C"
    / "results_fmnist_1_layer"
)

OUT = ROOT / "FinalFigs_manuscript"
OUT.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Figure-2 grid
# ------------------------------------------------------------

DENDS = [2, 4, 8, 16, 32, 64]
SOMATA = [32, 64, 128, 256, 512]

# Keep all comparisons balanced.
TRIALS = [1, 2, 3, 4, 5]

SIGMA = 0.0


# ============================================================
# MODELS
#
# This is the ONLY place that needs editing if we add another
# dANN-C condition later.
# ============================================================

C_MODELS = {

    "dANN-C 10%, λ=0.10":
        C_ROOT
        / "dend_ann_local_rfs_C_frac10_lam0p1",

    "dANN-C 20%, λ=0.10":
        C_ROOT
        / "dend_ann_local_rfs_C",

    "dANN-C 30%, λ=0.10":
        C_ROOT
        / "dend_ann_local_rfs_C_frac30_lam0p1",

    "dANN-C 40%, λ=0.10":
        C_ROOT
        / "dend_ann_local_rfs_C_frac40_lam0p1",

    "dANN-C 20%, λ=0.05":
        C_ROOT
        / "dend_ann_local_rfs_C_frac20_lam0p05",

    "dANN-C 20%, λ=0.20":
        C_ROOT
        / "dend_ann_local_rfs_C_frac20_lam0p2",
}


MODEL_ORDER = [

    "vANN",
    "dANN-LRF",

    "dANN-C 10%, λ=0.10",
    "dANN-C 20%, λ=0.10",
    "dANN-C 30%, λ=0.10",
    "dANN-C 40%, λ=0.10",

    "dANN-C 20%, λ=0.05",
    "dANN-C 20%, λ=0.20",
]


# ============================================================
# COLORS
#
# Fraction manipulation = reds/purples
# Lambda manipulation   = blues
# Original architectures retain recognizable colors.
# ============================================================

PALETTE = {

    "vANN":
        "#8d8d8d",

    "dANN-LRF":
        "#ff9f9b",

    # fraction series
    "dANN-C 10%, λ=0.10":
        "#d8b4d8",

    "dANN-C 20%, λ=0.10":
        "#8c2d8c",

    "dANN-C 30%, λ=0.10":
        "#b2182b",

    "dANN-C 40%, λ=0.10":
        "#67001f",

    # lambda series
    "dANN-C 20%, λ=0.05":
        "#67a9cf",

    "dANN-C 20%, λ=0.20":
        "#2166ac",
}


# ============================================================
# TRAINABLE PARAMETER COUNT
#
# Coupling is fixed and therefore adds NO trainable parameters.
# ============================================================

def num_trainable_params(
    D,
    S,
    n_syn=16,
    n_class=10,
):
    return (
        (n_syn + 1) * D * S
        + (D + 1) * S
        + (S + 1) * n_class
    )


# ============================================================
# LOAD ORIGINAL vANN + dANN-LRF
# ============================================================

print("\n============================================")
print("LOADING ORIGINAL MODELS")
print("============================================")


with open(ORIGINAL_FILE, "rb") as f:
    original = pickle.load(f)


df_original = fix_names(
    original["testing"]
).copy()


# Original test_acc stored as fraction
df_original["test_acc"] *= 100


# Keep only models needed here
df_original = df_original[
    df_original["model"].isin(
        [
            "vANN",
            "dANN-LRF",
        ]
    )
].copy()


# Keep only trials 1-5 if trial information is present
if "trial" in df_original.columns:

    df_original = df_original[
        df_original["trial"].isin(TRIALS)
    ].copy()


print(
    df_original
    .groupby("model")
    .size()
)


# ============================================================
# LOAD dANN-C RESULTS
# ============================================================

pattern = re.compile(
    r"results_sigma_0\.0"
    r"_trial_(\d+)"
    r"_dends_(\d+)"
    r"_soma_(\d+)\.pkl"
)


rows = []


for model_label, folder in C_MODELS.items():

    print("\n============================================")
    print(model_label)
    print("============================================")
    print(folder)


    if not folder.exists():

        print("WARNING: folder does not exist.")
        continue


    n_loaded = 0


    for file in folder.glob(
        "results_sigma_0.0_trial_*_dends_*_soma_*.pkl"
    ):

        match = pattern.fullmatch(
            file.name
        )

        if not match:
            continue


        trial, D, S = map(
            int,
            match.groups(),
        )


        # -----------------------------------------------
        # Restrict to Figure-2 grid
        # -----------------------------------------------

        if trial not in TRIALS:
            continue

        if D not in DENDS:
            continue

        if S not in SOMATA:
            continue


        with open(file, "rb") as f:
            result = pickle.load(f)


        rows.append({

            "model":
                model_label,

            "trial":
                trial,

            "num_dends":
                D,

            "num_soma":
                S,

            "test_loss":
                float(
                    np.asarray(
                        result["test_loss"]
                    ).squeeze()
                ),

            "test_acc":
                float(
                    np.asarray(
                        result["test_acc"]
                    ).squeeze()
                ) * 100,

            "trainable_params":
                num_trainable_params(
                    D,
                    S,
                ),
        })


        n_loaded += 1


    print(
        f"Loaded: {n_loaded}/150 runs"
    )


    if n_loaded != 150:

        print(
            "WARNING: this condition is not complete."
        )


df_C = pd.DataFrame(
    rows
)


# ============================================================
# COMPLETENESS CHECK
# ============================================================

print("\n============================================")
print("dANN-C COMPLETENESS")
print("============================================")


for model_label in C_MODELS:

    d = df_C[
        df_C["model"] == model_label
    ]


    print(
        f"{model_label:25s}: "
        f"{len(d):3d}/150"
    )


# ============================================================
# MORE DETAILED MISSING-RUN CHECK
# ============================================================

print("\n============================================")
print("MISSING CONFIGURATIONS")
print("============================================")


for model_label in C_MODELS:

    d = df_C[
        df_C["model"] == model_label
    ]


    existing = set(
        zip(
            d["trial"],
            d["num_dends"],
            d["num_soma"],
        )
    )


    missing = []


    for D in DENDS:

        for S in SOMATA:

            for trial in TRIALS:

                key = (
                    trial,
                    D,
                    S,
                )

                if key not in existing:

                    missing.append(
                        key
                    )


    if not missing:

        print(
            f"{model_label}: COMPLETE"
        )

    else:

        print(
            f"{model_label}: "
            f"{len(missing)} missing"
        )


# ============================================================
# COMBINE DATA
# ============================================================

df = pd.concat(
    [
        df_original,
        df_C,
    ],
    ignore_index=True,
)


# ============================================================
# PERFORMANCE SUMMARY
#
# Mean of each architecture across its five seeds.
# ============================================================

architecture_summary = (

    df.groupby(
        [
            "model",
            "num_dends",
            "num_soma",
            "trainable_params",
        ],
        as_index=False,
    )

    .agg(

        test_acc_mean=(
            "test_acc",
            "mean",
        ),

        test_acc_sd=(
            "test_acc",
            "std",
        ),

        test_loss_mean=(
            "test_loss",
            "mean",
        ),

        test_loss_sd=(
            "test_loss",
            "std",
        ),

        n=(
            "test_acc",
            "count",
        ),
    )
)


# ============================================================
# PRINT TOP ARCHITECTURE OF EACH MODEL
#
# Descriptive only:
# highest mean test accuracy.
# ============================================================

print("\n============================================")
print("HIGHEST-ACCURACY ARCHITECTURE PER MODEL")
print("============================================")


for model in MODEL_ORDER:

    d = architecture_summary[
        architecture_summary["model"]
        == model
    ]


    if d.empty:
        continue


    best = d.loc[
        d["test_acc_mean"].idxmax()
    ]


    print(
        f"{model:25s} "
        f"D={int(best.num_dends):2d} "
        f"S={int(best.num_soma):3d} | "
        f"acc={best.test_acc_mean:.3f}% | "
        f"loss={best.test_loss_mean:.4f} | "
        f"n={int(best.n)}"
    )


# ============================================================
# vANN REFERENCE LINES
#
# Same logic as original Figure 2:
# use mean performance for each vANN architecture.
# ============================================================

vann_arch = architecture_summary[
    architecture_summary["model"]
    == "vANN"
].copy()


vann_min_loss_row = vann_arch.loc[
    vann_arch[
        "test_loss_mean"
    ].idxmin()
]


vann_max_acc_row = vann_arch.loc[
    vann_arch[
        "test_acc_mean"
    ].idxmax()
]


vann_min_loss = (
    vann_min_loss_row[
        "test_loss_mean"
    ]
)

vann_loss_params = (
    vann_min_loss_row[
        "trainable_params"
    ]
)


vann_max_acc = (
    vann_max_acc_row[
        "test_acc_mean"
    ]
)

vann_acc_params = (
    vann_max_acc_row[
        "trainable_params"
    ]
)


print("\n============================================")
print("vANN REFERENCES")
print("============================================")

print(
    f"Minimum loss: "
    f"{vann_min_loss:.4f}"
)

print(
    f"Maximum accuracy: "
    f"{vann_max_acc:.3f}%"
)


# ============================================================
# STYLE
# ============================================================

sns.set_style(
    "white"
)

plt.rcParams.update(
    my_style()
)


# ============================================================
# FIGURE
#
# Same B-E organization as Figure 2.
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9.0, 7.5),
)


# ============================================================
# PANEL B
# TEST LOSS vs TRAINABLE PARAMETERS
# ============================================================

ax = axes[0, 0]


sns.lineplot(
    data=df,
    x="trainable_params",
    y="test_loss",
    hue="model",
    hue_order=MODEL_ORDER,
    style="model",
    style_order=MODEL_ORDER,
    markers=True,
    dashes=False,
    palette=PALETTE,
    errorbar=("ci", 95),
    legend=False,
    ax=ax,
)


ax.axhline(
    vann_min_loss,
    linestyle="--",
    color="black",
    linewidth=1.2,
)


ax.axvline(
    vann_loss_params,
    linestyle="--",
    color="black",
    linewidth=1.2,
)


ax.set_xscale(
    "log"
)

ax.set_xlabel(
    "Trainable parameters"
)

ax.set_ylabel(
    "Test loss"
)


# ============================================================
# PANEL C
# TEST ACCURACY vs TRAINABLE PARAMETERS
# ============================================================

ax = axes[0, 1]


sns.lineplot(
    data=df,
    x="trainable_params",
    y="test_acc",
    hue="model",
    hue_order=MODEL_ORDER,
    style="model",
    style_order=MODEL_ORDER,
    markers=True,
    dashes=False,
    palette=PALETTE,
    errorbar=("ci", 95),
    ax=ax,
)


ax.axhline(
    vann_max_acc,
    linestyle="--",
    color="black",
    linewidth=1.2,
)


ax.axvline(
    vann_acc_params,
    linestyle="--",
    color="black",
    linewidth=1.2,
)


ax.set_xscale(
    "log"
)

ax.set_xlabel(
    "Trainable parameters"
)

ax.set_ylabel(
    "Test accuracy (%)"
)


# ============================================================
# PANELS D/E
#
# Same restriction as original Figure 2:
# only S = 256 and 512.
#
# vANN excluded because D does not carry the same biological
# interpretation in this panel.
# ============================================================

df_de = df[
    (df["num_soma"] > 200)
    &
    (df["model"] != "vANN")
].copy()


# ============================================================
# PANEL D
# TEST LOSS vs DENDRITES PER SOMA
# ============================================================

ax = axes[1, 0]


sns.lineplot(
    data=df_de,
    x="num_dends",
    y="test_loss",
    hue="model",
    hue_order=[
        m
        for m in MODEL_ORDER
        if m != "vANN"
    ],
    style="num_soma",
    markers=True,
    markersize=6,
    palette=PALETTE,
    errorbar=("ci", 95),
    legend=False,
    ax=ax,
)


ax.axhline(
    vann_min_loss,
    linestyle="--",
    color="black",
    linewidth=1.2,
)


ax.set_xscale(
    "log",
    base=2,
)

ax.set_xticks(
    DENDS
)

ax.set_xticklabels(
    DENDS
)

ax.set_xlabel(
    "Dendrites per soma"
)

ax.set_ylabel(
    "Test loss"
)


# ============================================================
# PANEL E
# TEST ACCURACY vs DENDRITES PER SOMA
# ============================================================

ax = axes[1, 1]


sns.lineplot(
    data=df_de,
    x="num_dends",
    y="test_acc",
    hue="model",
    hue_order=[
        m
        for m in MODEL_ORDER
        if m != "vANN"
    ],
    style="num_soma",
    markers=True,
    markersize=6,
    palette=PALETTE,
    errorbar=("ci", 95),
    ax=ax,
)


ax.axhline(
    vann_max_acc,
    linestyle="--",
    color="black",
    linewidth=1.2,
)


ax.set_xscale(
    "log",
    base=2,
)

ax.set_xticks(
    DENDS
)

ax.set_xticklabels(
    DENDS
)

ax.set_xlabel(
    "Dendrites per soma"
)

ax.set_ylabel(
    "Test accuracy (%)"
)


# ============================================================
# PANEL LABELS
# ============================================================

for ax, label in zip(
    axes.flat,
    ["b", "c", "d", "e"],
):

    ax.text(
        -0.15,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
    )


# ============================================================
# CLEAN LEGEND
# ============================================================

# Keep the complete model legend only in panel C.
#
# Panel E's automatically generated legend contains both
# model and soma-style information and can get huge.

if axes[1, 1].get_legend() is not None:

    axes[1, 1].get_legend().remove()


# ============================================================
# FORMAT
# ============================================================

sns.despine()

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

PDF = (
    OUT
    / "figure_2_all_dANNC_models.pdf"
)

SVG = (
    OUT
    / "figure_2_all_dANNC_models.svg"
)

CSV = (
    OUT
    / "figure_2_all_dANNC_models_summary.csv"
)


fig.savefig(
    PDF,
    bbox_inches="tight",
)

fig.savefig(
    SVG,
    bbox_inches="tight",
    dpi=600,
)

architecture_summary.to_csv(
    CSV,
    index=False,
)


plt.close(fig)


print("\n============================================")
print("SAVED")
print("============================================")

print(PDF)
print(SVG)
print(CSV)