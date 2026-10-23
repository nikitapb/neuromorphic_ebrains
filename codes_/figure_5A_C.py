# figure_5A_C.py

import os
import pickle
import pathlib
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from scipy.stats import skew, kurtosis
import keras

# Import only to register the custom Keras layer when loading dANN-C models
from opt import DendriticCoupling

from plotting_functions import my_style, draw_text_metrics


# ---------------------------------------------------------------------
# PATHS / SETTINGS
# ---------------------------------------------------------------------

ROOT = pathlib.Path.home() / "dendritic_anns"

WEIGHTS_FILE = ROOT / "DATA/weights_best_models.pkl"

C_DIR = (
    ROOT
    / "DATA_C/results_fmnist_1_layer/dend_ann_local_rfs_C"
)

OUT = ROOT / "FinalFigs_manuscript"
OUT.mkdir(exist_ok=True)

D_C = 8
S_C = 512
N_TRIALS = 5
SIGMA = 0.0


# ---------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------

sns.set_style("white")
plt.rcParams.update(my_style())


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def remove_zeros(x):
    x = np.asarray(x).ravel()
    return x[x != 0]


def get_layer_weights(model, layer_name):
    layer = model.get_layer(layer_name)
    return layer.get_weights()[0]


# ---------------------------------------------------------------------
# LOAD ORIGINAL PRECOMPUTED WEIGHTS
# ---------------------------------------------------------------------

with open(WEIGHTS_FILE, "rb") as f:
    weights = pickle.load(f)

w_dend = weights["dendrites"]
w_soma = weights["soma"]
w_out = weights["output"]


# Original Figure 5A models
models = [
    "dend_ann_random",
    "dend_ann_local_rfs",
    "dend_ann_global_rfs",
    "vanilla_ann",
]

model_names = [
    "dANN-R",
    "dANN-LRF",
    "dANN-GRF",
    "vANN",
]


# ---------------------------------------------------------------------
# LOAD dANN-C: D=8, S=512, FIVE TRIALS
# ---------------------------------------------------------------------

C_dend = []
C_soma = []
C_out = []

for trial in range(1, N_TRIALS + 1):

    postfix = (
        f"sigma_{SIGMA}_trial_{trial}"
        f"_dends_{D_C}_soma_{S_C}"
    )

    fname = C_DIR / f"model_{postfix}.keras"

    print(f"Loading {fname.name}")

    model = keras.models.load_model(fname)

    C_dend.append(
        remove_zeros(
            get_layer_weights(model, "dend_1")
        )
    )

    C_soma.append(
        remove_zeros(
            get_layer_weights(model, "soma_1")
        )
    )

    C_out.append(
        remove_zeros(
            get_layer_weights(model, "output")
        )
    )

    keras.backend.clear_session()


C_dend = np.concatenate(C_dend)
C_soma = np.concatenate(C_soma)
C_out = np.concatenate(C_out)

print("\ndANN-C pooled weights:")
print("dendritic:", len(C_dend))
print("somatic:  ", len(C_soma))
print("output:   ", len(C_out))


# Add dANN-C to precomputed dictionaries
w_dend["dend_ann_local_rfs_C"] = C_dend
w_soma["dend_ann_local_rfs_C"] = C_soma
w_out["dend_ann_local_rfs_C"] = C_out

models.append("dend_ann_local_rfs_C")
model_names.append("dANN-C")


# ---------------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------------

palette_dend = [
    "#8de5a1",   # dANN-R
    "#ff9f9b",   # dANN-LRF
    "#a1c9f4",   # dANN-GRF
    "#b5b5ac",   # vANN
    "#9b59b6",   # dANN-C
]

palette_soma = [
    "#409140",
    "#e06666",
    "#7abacc",
    "#8d8d8d",
    "#6c3483",
]


# ---------------------------------------------------------------------
# FIGURE — PANEL 5A ONLY
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    5,
    figsize=(9.0, 5.6),
    sharex=True,
    sharey=True,
)

trans = mtransforms.ScaledTranslation(
    -20 / 72,
    7 / 72,
    fig.dpi_scale_trans,
)

axes[0, 0].text(
    0,
    1,
    "a",
    transform=axes[0, 0].transAxes + trans,
    fontsize="large",
    va="bottom",
)

xloc = 0.97
yloc = 0.97


# ---------------------------------------------------------------------
# ROW 1 — INPUT -> DENDRITE WEIGHTS
# ---------------------------------------------------------------------

for i, model_type in enumerate(models):

    ax = axes[0, i]

    vals = np.asarray(
        w_dend[model_type]
    ).flatten()

    vals = vals[vals != 0]

    sns.histplot(
        vals,
        bins=20,
        kde=True,
        stat="probability",
        color=palette_dend[i],
        ax=ax,
    )

    ax.set_title(model_names[i])
    ax.set_ylabel(
        "probability" if i == 0 else ""
    )

    ax.set_xticks([])
    ax.set_xticklabels([])

    draw_text_metrics(
        ax,
        xloc,
        yloc,
        kurtosis(vals),
        "Kurtosis",
    )

    draw_text_metrics(
        ax,
        xloc,
        yloc - 0.2,
        skew(vals),
        "Skewness",
    )

    draw_text_metrics(
        ax,
        xloc,
        yloc - 0.4,
        np.ptp(vals),
        "Range",
    )


# ---------------------------------------------------------------------
# ROW 2 — DENDRITE -> SOMA WEIGHTS
# ---------------------------------------------------------------------

for i, model_type in enumerate(models):

    ax = axes[1, i]

    vals = np.asarray(
        w_soma[model_type]
    ).flatten()

    vals = vals[vals != 0]

    sns.histplot(
        vals,
        bins=20,
        kde=True,
        stat="probability",
        color=palette_soma[i],
        ax=ax,
    )

    ax.set_xlabel("weights")
    ax.set_ylabel(
        "probability" if i == 0 else ""
    )

    ax.set_xticks([-2.5, 0, 2.5])

    draw_text_metrics(
        ax,
        xloc,
        yloc,
        kurtosis(vals),
        "Kurtosis",
    )

    draw_text_metrics(
        ax,
        xloc,
        yloc - 0.2,
        skew(vals),
        "Skewness",
    )

    draw_text_metrics(
        ax,
        xloc,
        yloc - 0.4,
        np.ptp(vals),
        "Range",
    )


# ---------------------------------------------------------------------
# ROW 3 — SOMA -> OUTPUT WEIGHTS
# ---------------------------------------------------------------------

for i, model_type in enumerate(models):

    ax = axes[2, i]

    vals = np.asarray(
        w_out[model_type]
    ).flatten()

    vals = vals[vals != 0]

    sns.histplot(
        vals,
        bins=20,
        kde=True,
        stat="probability",
        color=palette_soma[i],
        ax=ax,
    )

    ax.set_xlabel("weights")
    ax.set_ylabel(
        "probability" if i == 0 else ""
    )

    ax.set_xticks([-2.5, 0, 2.5])

    draw_text_metrics(
        ax,
        xloc,
        yloc,
        kurtosis(vals),
        "Kurtosis",
    )

    draw_text_metrics(
        ax,
        xloc,
        yloc - 0.2,
        skew(vals),
        "Skewness",
    )

    draw_text_metrics(
        ax,
        xloc,
        yloc - 0.4,
        np.ptp(vals),
        "Range",
    )


# ---------------------------------------------------------------------
# FORMAT / SAVE
# ---------------------------------------------------------------------

fig.tight_layout()

svg = OUT / "figure_5A_with_dANNC.svg"
pdf = OUT / "figure_5A_with_dANNC.pdf"

fig.savefig(
    svg,
    bbox_inches="tight",
    dpi=600,
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

plt.close(fig)

print("\nSaved:")
print(svg)
print(pdf)
