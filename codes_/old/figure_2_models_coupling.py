# figure_2_models_coupling.py

import re
import pickle
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_functions import fix_names, calculate_best_model


# ============================================================
# SETTINGS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

OLD_FILE = (
    ROOT
    / "DATA/results_fmnist_1_layer/output_all_final.pkl"
)

DATA_C_ROOT = (
    ROOT
    / "DATA_C/results_fmnist_1_layer"
)

OUT = ROOT / "FinalFigs_manuscript"
OUT.mkdir(exist_ok=True)

SAVE = OUT / "figure_2_dANNC_coupling_comparison.pdf"

# Focused architecture used for the coupling experiment
D = 8
S = 512

# IMPORTANT:
# use only trials 1-5 for ALL coupling conditions
TRIALS = [1, 2, 3, 4, 5]


# ============================================================
# TRAINABLE PARAMETERS
#
# Coupling adds no trainable parameters.
# ============================================================

def num_trainable_params(D, S, n_syn=16, n_class=10):

    return (
        (n_syn + 1) * D * S
        + (D + 1) * S
        + (S + 1) * n_class
    )


# ============================================================
# LOAD ORIGINAL PAPER DATA
# ============================================================

with open(OLD_FILE, "rb") as f:
    results = pickle.load(f)


df_old = fix_names(
    results["testing"]
).copy()

df_old["test_acc"] *= 100


# ------------------------------------------------------------
# Keep only:
#
#   vANN
#   dANN-LRF
#
# These retain the complete original Figure-2 architecture grid.
# ------------------------------------------------------------

df_old = df_old[
    df_old["model"].isin(
        [
            "vANN",
            "dANN-LRF",
        ]
    )
].copy()


# ============================================================
# ORIGINAL vANN BENCHMARK
#
# Same reference logic as Figure 2:
# minimum loss and maximum accuracy across the vANN grid.
# ============================================================

vann_metrics = calculate_best_model(
    df_old[
        df_old["model"] == "vANN"
    ]
)

vann_min_loss = np.nanmin(
    vann_metrics[:, 0]
)

vann_max_acc = np.nanmax(
    vann_metrics[:, 1]
)

vann_loss_params = (
    vann_metrics[
        np.nanargmin(
            vann_metrics[:, 0]
        )
    ][-1]
)

vann_acc_params = (
    vann_metrics[
        np.nanargmax(
            vann_metrics[:, 1]
        )
    ][-1]
)


print("\n====================================")
print("vANN REFERENCE")
print("====================================")

print(
    f"Minimum loss: "
    f"{vann_min_loss:.4f}"
)

print(
    f"Maximum accuracy: "
    f"{vann_max_acc:.3f}%"
)


# ============================================================
# dANN-C CONDITIONS
# ============================================================

conditions = {

    # --------------------------------------------------------
    # Original dANN-C reference
    # --------------------------------------------------------

    "dANN-C\n20%, λ=0.10": (
        DATA_C_ROOT
        / "dend_ann_local_rfs_C"
    ),

    # --------------------------------------------------------
    # Coupled-fraction manipulation
    # --------------------------------------------------------

    "dANN-C\n30%, λ=0.10": (
        DATA_C_ROOT
        / "dend_ann_local_rfs_C_frac30_lam0p1"
    ),

    "dANN-C\n40%, λ=0.10": (
        DATA_C_ROOT
        / "dend_ann_local_rfs_C_frac40_lam0p1"
    ),

    # --------------------------------------------------------
    # Coupling-strength manipulation
    # --------------------------------------------------------

    "dANN-C\n20%, λ=0.05": (
        DATA_C_ROOT
        / "dend_ann_local_rfs_C_frac20_lam0p05"
    ),

    "dANN-C\n20%, λ=0.20": (
        DATA_C_ROOT
        / "dend_ann_local_rfs_C_frac20_lam0p2"
    ),
}


# ============================================================
# LOAD dANN-C RESULTS
# ============================================================

rows = []


pattern = re.compile(
    r"results_sigma_0\.0_trial_(\d+)"
    r"_dends_(\d+)"
    r"_soma_(\d+)\.pkl"
)


for label, folder in conditions.items():

    print("\n------------------------------------")
    print(label.replace("\n", " "))
    print("------------------------------------")
    print(folder)

    if not folder.exists():

        raise FileNotFoundError(
            f"\nFolder not found:\n{folder}"
        )


    files_found = 0


    for file in folder.glob(
        "results_sigma_0.0_trial_*"
        "_dends_8_soma_512.pkl"
    ):

        m = pattern.fullmatch(
            file.name
        )

        if not m:
            continue


        trial, dends, soma = map(
            int,
            m.groups(),
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # reference currently has >5 trials,
        # but we only use 1-5 here.
        # ----------------------------------------------------

        if trial not in TRIALS:
            continue


        if (
            dends != D
            or soma != S
        ):
            continue


        with open(file, "rb") as f:
            out = pickle.load(f)


        rows.append({

            "model": label,

            "trial": trial,

            "num_dends": dends,

            "num_soma": soma,

            "test_loss":
                float(
                    np.asarray(
                        out["test_loss"]
                    ).squeeze()
                ),

            "test_acc":
                float(
                    np.asarray(
                        out["test_acc"]
                    ).squeeze()
                ) * 100,

            "trainable_params":
                num_trainable_params(
                    dends,
                    soma,
                ),
        })


        files_found += 1


    print(
        f"Trials loaded: {files_found}"
    )


# ============================================================
# DATAFRAME
# ============================================================

df_C = pd.DataFrame(rows)


print("\n====================================")
print("dANN-C RUNS LOADED")
print("====================================")

print(
    df_C.groupby("model")["trial"]
        .agg(
            [
                "count",
                "min",
                "max",
            ]
        )
)


# ============================================================
# SAFETY CHECK
#
# Exactly 5 trials per dANN-C condition.
# ============================================================

for label in conditions:

    n = (
        df_C[
            df_C["model"] == label
        ]["trial"]
        .nunique()
    )

    if n != 5:

        raise RuntimeError(
            f"{label}: expected 5 trials, "
            f"found {n}"
        )


# ============================================================
# PRINT PERFORMANCE SUMMARY
# ============================================================

summary = (

    df_C

    .groupby("model")

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
    )
)


print("\n====================================")
print("COUPLING PERFORMANCE SUMMARY")
print("====================================")

print(
    summary.round(4)
)


# ============================================================
# COMBINE ORIGINAL MODELS + dANN-C
# ============================================================

df = pd.concat(
    [
        df_old,
        df_C,
    ],
    ignore_index=True,
)


# ============================================================
# MODEL ORDER
# ============================================================

order = [

    "vANN",

    "dANN-LRF",

    "dANN-C\n20%, λ=0.10",

    "dANN-C\n30%, λ=0.10",

    "dANN-C\n40%, λ=0.10",

    "dANN-C\n20%, λ=0.05",

    "dANN-C\n20%, λ=0.20",
]


# ============================================================
# COLORS
# ============================================================

palette = {

    "vANN":
        "#8d8d8d",

    "dANN-LRF":
        "#ff9f9b",

    # Reference coupling
    "dANN-C\n20%, λ=0.10":
        "#7f0000",

    # Fraction experiment
    "dANN-C\n30%, λ=0.10":
        "#b2182b",

    "dANN-C\n40%, λ=0.10":
        "#d6604d",

    # Lambda experiment
    "dANN-C\n20%, λ=0.05":
        "#762a83",

    "dANN-C\n20%, λ=0.20":
        "#1b7837",
}


# ============================================================
# STYLE
# ============================================================

sns.set_style("white")

plt.rcParams.update({

    "font.family":
        "Arial",

    "axes.spines.top":
        False,

    "axes.spines.right":
        False,
})


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    2,
    2,
    figsize=(8, 7),
)


# ============================================================
# B — TEST LOSS vs TRAINABLE PARAMETERS
# ============================================================

sns.lineplot(

    data=df,

    x="trainable_params",

    y="test_loss",

    hue="model",

    hue_order=order,

    style="model",

    style_order=order,

    markers=True,

    dashes=False,

    palette=palette,

    errorbar=("ci", 95),

    legend=False,

    ax=ax[0, 0],
)


ax[0, 0].axhline(
    vann_min_loss,
    ls="--",
    c="k",
    lw=1,
)


ax[0, 0].axvline(
    vann_loss_params,
    ls="--",
    c="k",
    lw=1,
)


ax[0, 0].set_xscale(
    "log"
)

ax[0, 0].set_xlabel(
    "Trainable parameters"
)

ax[0, 0].set_ylabel(
    "Test loss"
)


# ============================================================
# C — TEST ACCURACY vs TRAINABLE PARAMETERS
# ============================================================

sns.lineplot(

    data=df,

    x="trainable_params",

    y="test_acc",

    hue="model",

    hue_order=order,

    style="model",

    style_order=order,

    markers=True,

    dashes=False,

    palette=palette,

    errorbar=("ci", 95),

    ax=ax[0, 1],
)


ax[0, 1].axhline(
    vann_max_acc,
    ls="--",
    c="k",
    lw=1,
)


ax[0, 1].axvline(
    vann_acc_params,
    ls="--",
    c="k",
    lw=1,
)


ax[0, 1].set_xscale(
    "log"
)

ax[0, 1].set_xlabel(
    "Trainable parameters"
)

ax[0, 1].set_ylabel(
    "Test accuracy (%)"
)


# ============================================================
# D / E
#
# Original Figure-2 logic:
# S > 200.
#
# For dANN-C, we currently have only D=8 / S=512 for these
# coupling conditions, so each condition appears as one point.
#
# dANN-LRF retains its complete grid.
# ============================================================

df_ds = df[
    (df["num_soma"] > 200)
    &
    (df["model"] != "vANN")
].copy()


# ============================================================
# D — TEST LOSS vs DENDRITES
# ============================================================

sns.lineplot(

    data=df_ds,

    x="num_dends",

    y="test_loss",

    hue="model",

    hue_order=order,

    style="num_soma",

    markers=True,

    markersize=7,

    palette=palette,

    errorbar=("ci", 95),

    legend=False,

    ax=ax[1, 0],
)


ax[1, 0].axhline(
    vann_min_loss,
    ls="--",
    c="k",
    lw=1,
)


ax[1, 0].set_xscale(
    "log",
    base=2,
)

ax[1, 0].set_xlabel(
    "Dendrites per soma"
)

ax[1, 0].set_ylabel(
    "Test loss"
)


# ============================================================
# E — TEST ACCURACY vs DENDRITES
# ============================================================

sns.lineplot(

    data=df_ds,

    x="num_dends",

    y="test_acc",

    hue="model",

    hue_order=order,

    style="num_soma",

    markers=True,

    markersize=7,

    palette=palette,

    errorbar=("ci", 95),

    ax=ax[1, 1],
)


ax[1, 1].axhline(
    vann_max_acc,
    ls="--",
    c="k",
    lw=1,
)


ax[1, 1].set_xscale(
    "log",
    base=2,
)

ax[1, 1].set_xlabel(
    "Dendrites per soma"
)

ax[1, 1].set_ylabel(
    "Test accuracy (%)"
)


# ============================================================
# PANEL LABELS
# ============================================================

for a, label in zip(
    ax.flat,
    [
        "b",
        "c",
        "d",
        "e",
    ],
):

    a.text(
        -0.15,
        1.05,
        label,

        transform=a.transAxes,

        fontsize=13,

        fontweight="bold",
    )


# ============================================================
# FORMAT
# ============================================================

sns.despine()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    SAVE,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    SAVE.with_suffix(".svg"),
    bbox_inches="tight",
)


plt.close(fig)


print("\n====================================")
print("SAVED")
print("====================================")

print(SAVE)
print(SAVE.with_suffix(".svg"))
