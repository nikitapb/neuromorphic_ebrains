# coupled_vs_uncoupled_weights_vs_D.py

import pathlib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras

from opt import DendriticCoupling
from plotting_functions import my_style


# ============================================================
# SETTINGS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

DANN_DIR = (
    ROOT
    / "DATA/results_fmnist_1_layer/dend_ann_local_rfs"
)

DANNC_DIR = (
    ROOT
    / "DATA_C/results_fmnist_1_layer/dend_ann_local_rfs_C"
)

OUT = ROOT / "FinalFigs_manuscript"
OUT.mkdir(exist_ok=True)

S = 512
D_VALUES = [2, 4, 8, 16, 32, 64]
TRIALS = [1, 2, 3, 4, 5]

SIGMA = 0.0

COUPLING_FRACTION = 0.20
N_COUPLED = int(round(S * COUPLING_FRACTION))

LAYERS = [
    "Input → dendrite",
    "Dendrite → soma",
    "Soma → output",
]

GROUPS = [
    "dANN",
    "dANN-C coupled",
    "dANN-C uncoupled",
]

COLORS = {
    "dANN": "#d95f5f",
    "dANN-C coupled": "#7f0000",
    "dANN-C uncoupled": "#777777",
}

plt.rcParams.update(my_style())


# ============================================================
# HELPERS
# ============================================================

def get_model_file(folder, trial, D, S):

    return folder / (
        f"model_sigma_{SIGMA}"
        f"_trial_{trial}"
        f"_dends_{D}"
        f"_soma_{S}.keras"
    )


def nonzero(arr):
    """
    Flatten and remove structural zero weights.
    """
    arr = np.asarray(arr).ravel()
    return arr[arr != 0]


def summarize_weights(weights):

    w = nonzero(weights)

    pos = w[w > 0]
    neg = w[w < 0]

    return {
        "n_weights": len(w),

        "total_abs_weight": np.sum(np.abs(w)),
        "mean_abs_weight": np.mean(np.abs(w)),

        "fraction_positive": len(pos) / len(w),
        "fraction_negative": len(neg) / len(w),

        "mean_positive": (
            np.mean(pos)
            if len(pos) > 0
            else np.nan
        ),

        "mean_negative": (
            np.mean(neg)
            if len(neg) > 0
            else np.nan
        ),
    }


def append_row(
    rows,
    model,
    group,
    trial,
    D,
    layer,
    weights,
):

    stats = summarize_weights(weights)

    rows.append({
        "model": model,
        "group": group,
        "trial": trial,
        "D": D,
        "S": S,
        "layer": layer,
        **stats,
    })


def mean_sem(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    mean = np.mean(values)

    if len(values) > 1:
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
    else:
        sem = 0

    return mean, sem


# ============================================================
# EXTRACT WEIGHTS
# ============================================================

rows = []


for D in D_VALUES:

    print("\n" + "=" * 60)
    print(f"D = {D}, S = {S}")
    print("=" * 60)

    for trial in TRIALS:

        # ====================================================
        # CONTROL dANN
        # ====================================================

        f_dann = get_model_file(
            DANN_DIR,
            trial,
            D,
            S,
        )

        if not f_dann.exists():
            raise FileNotFoundError(
                f"Missing dANN model:\n{f_dann}"
            )

        print(
            f"dANN   | D={D:2d} | "
            f"S={S} | trial={trial}"
        )

        model = keras.models.load_model(
            f_dann,
            compile=False,
        )

        W_input = (
            model
            .get_layer("dend_1")
            .get_weights()[0]
        )

        W_soma = (
            model
            .get_layer("soma_1")
            .get_weights()[0]
        )

        W_output = (
            model
            .get_layer("output")
            .get_weights()[0]
        )


        append_row(
            rows,
            "dANN",
            "dANN",
            trial,
            D,
            "Input → dendrite",
            W_input,
        )

        append_row(
            rows,
            "dANN",
            "dANN",
            trial,
            D,
            "Dendrite → soma",
            W_soma,
        )

        append_row(
            rows,
            "dANN",
            "dANN",
            trial,
            D,
            "Soma → output",
            W_output,
        )

        del model
        keras.backend.clear_session()


        # ====================================================
        # dANN-C
        # ====================================================

        f_c = get_model_file(
            DANNC_DIR,
            trial,
            D,
            S,
        )

        if not f_c.exists():
            raise FileNotFoundError(
                f"Missing dANN-C model:\n{f_c}"
            )

        print(
            f"dANN-C | D={D:2d} | "
            f"S={S} | trial={trial}"
        )

        model = keras.models.load_model(
            f_c,
            compile=False,
        )

        W_input = (
            model
            .get_layer("dend_1")
            .get_weights()[0]
        )

        W_soma = (
            model
            .get_layer("soma_1")
            .get_weights()[0]
        )

        W_output = (
            model
            .get_layer("output")
            .get_weights()[0]
        )


        # ====================================================
        # SPLIT dANN-C:
        #
        # first 20% somata = coupled
        # remaining 80%   = uncoupled
        # ====================================================

        coupled_input = []
        uncoupled_input = []

        coupled_soma = []
        uncoupled_soma = []

        coupled_output = []
        uncoupled_output = []


        for soma_idx in range(S):

            coupled = (
                soma_idx < N_COUPLED
            )

            d0 = soma_idx * D
            d1 = (soma_idx + 1) * D


            # -----------------------------------------------
            # INPUT -> DENDRITE
            # -----------------------------------------------

            w = nonzero(
                W_input[:, d0:d1]
            )

            if coupled:
                coupled_input.append(w)
            else:
                uncoupled_input.append(w)


            # -----------------------------------------------
            # DENDRITE -> SOMA
            # -----------------------------------------------

            w = nonzero(
                W_soma[d0:d1, soma_idx]
            )

            if coupled:
                coupled_soma.append(w)
            else:
                uncoupled_soma.append(w)


            # -----------------------------------------------
            # SOMA -> OUTPUT
            # -----------------------------------------------

            w = nonzero(
                W_output[soma_idx, :]
            )

            if coupled:
                coupled_output.append(w)
            else:
                uncoupled_output.append(w)


        coupled_input = np.concatenate(
            coupled_input
        )
        uncoupled_input = np.concatenate(
            uncoupled_input
        )

        coupled_soma = np.concatenate(
            coupled_soma
        )
        uncoupled_soma = np.concatenate(
            uncoupled_soma
        )

        coupled_output = np.concatenate(
            coupled_output
        )
        uncoupled_output = np.concatenate(
            uncoupled_output
        )


        # ====================================================
        # STORE COUPLED / UNCOUPLED
        # ====================================================

        append_row(
            rows,
            "dANN-C",
            "dANN-C coupled",
            trial,
            D,
            "Input → dendrite",
            coupled_input,
        )

        append_row(
            rows,
            "dANN-C",
            "dANN-C uncoupled",
            trial,
            D,
            "Input → dendrite",
            uncoupled_input,
        )


        append_row(
            rows,
            "dANN-C",
            "dANN-C coupled",
            trial,
            D,
            "Dendrite → soma",
            coupled_soma,
        )

        append_row(
            rows,
            "dANN-C",
            "dANN-C uncoupled",
            trial,
            D,
            "Dendrite → soma",
            uncoupled_soma,
        )


        append_row(
            rows,
            "dANN-C",
            "dANN-C coupled",
            trial,
            D,
            "Soma → output",
            coupled_output,
        )

        append_row(
            rows,
            "dANN-C",
            "dANN-C uncoupled",
            trial,
            D,
            "Soma → output",
            uncoupled_output,
        )

        del model
        keras.backend.clear_session()


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


print("\n\n")
print("=" * 70)
print("CHECKING EXTRACTED DATA")
print("=" * 70)

print(
    df.groupby(
        [
            "D",
            "layer",
            "group",
        ]
    )["trial"]
    .nunique()
)


# ============================================================
# IMPORTANT CHECK:
# PRINT VALUES THAT WILL ACTUALLY BE PLOTTED
# ============================================================

print("\n")
print("=" * 70)
print("MEAN FRACTION POSITIVE")
print("=" * 70)

print(
    df.groupby(
        [
            "layer",
            "group",
            "D",
        ]
    )["fraction_positive"]
    .mean()
    .round(4)
)


print("\n")
print("=" * 70)
print("MEAN POSITIVE STRENGTH")
print("=" * 70)

print(
    df.groupby(
        [
            "layer",
            "group",
            "D",
        ]
    )["mean_positive"]
    .mean()
    .round(4)
)


print("\n")
print("=" * 70)
print("MEAN NEGATIVE STRENGTH")
print("=" * 70)

print(
    df.groupby(
        [
            "layer",
            "group",
            "D",
        ]
    )["mean_negative"]
    .mean()
    .round(4)
)


# ============================================================
# WHOLE NETWORK dANN-C FOR TOTAL ABSOLUTE WEIGHT
# ============================================================

# Coupled and uncoupled parts are summed because together
# they constitute the complete dANN-C network.

df_c_total = (
    df[
        df["model"] == "dANN-C"
    ]
    .groupby(
        [
            "trial",
            "D",
            "layer",
        ],
        as_index=False,
    )["total_abs_weight"]
    .sum()
)

df_c_total["group"] = "dANN-C"


df_dann_total = (
    df[
        df["group"] == "dANN"
    ][
        [
            "trial",
            "D",
            "layer",
            "total_abs_weight",
        ]
    ]
    .copy()
)

df_dann_total["group"] = "dANN"


df_total = pd.concat(
    [
        df_dann_total,
        df_c_total,
    ],
    ignore_index=True,
)


# ============================================================
# MANUAL PLOTTING FUNCTION
# ============================================================

def plot_metric(
    ax,
    data,
    metric,
    groups,
    ylabel=None,
):

    x_positions = np.arange(
        len(D_VALUES)
    )

    for group in groups:

        means = []
        sems = []

        for D in D_VALUES:

            vals = data[
                (data["group"] == group)
                &
                (data["D"] == D)
            ][metric].values

            mean, sem = mean_sem(vals)

            means.append(mean)
            sems.append(sem)


        means = np.asarray(means)
        sems = np.asarray(sems)


        ax.errorbar(
            x_positions,
            means,
            yerr=sems,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=group,
            color=COLORS.get(
                group,
                None,
            ),
        )


        # Also plot each individual trial faintly
        for trial in TRIALS:

            trial_values = []

            for D in D_VALUES:

                temp = data[
                    (data["group"] == group)
                    &
                    (data["D"] == D)
                    &
                    (data["trial"] == trial)
                ]

                if len(temp) == 1:
                    trial_values.append(
                        temp.iloc[0][metric]
                    )
                else:
                    trial_values.append(
                        np.nan
                    )

            ax.plot(
                x_positions,
                trial_values,
                color=COLORS.get(
                    group,
                    None,
                ),
                alpha=0.15,
                linewidth=0.7,
            )


    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        D_VALUES
    )

    ax.set_xlabel(
        "Dendrites per soma"
    )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=4,
    ncols=3,
    figsize=(11, 12),
    sharex=True,
)


# ============================================================
# ROW 1:
# TOTAL ABSOLUTE WEIGHT
#
# WHOLE dANN
# vs
# WHOLE dANN-C
# ============================================================

for col, layer in enumerate(LAYERS):

    ax = axes[0, col]

    sub = df_total[
        df_total["layer"] == layer
    ]

    plot_metric(
        ax=ax,
        data=sub,
        metric="total_abs_weight",
        groups=[
            "dANN",
            "dANN-C",
        ],
        ylabel=(
            "Total absolute weight"
            if col == 0
            else None
        ),
    )

    ax.set_title(
        layer
    )


# ============================================================
# ROW 2:
# POSITIVE FRACTION
#
# dANN
# coupled dANN-C
# uncoupled dANN-C
# ============================================================

for col, layer in enumerate(LAYERS):

    ax = axes[1, col]

    sub = df[
        df["layer"] == layer
    ]

    plot_metric(
        ax=ax,
        data=sub,
        metric="fraction_positive",
        groups=GROUPS,
        ylabel=(
            "Fraction positive"
            if col == 0
            else None
        ),
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=0.8,
        color="black",
        alpha=0.4,
    )


# ============================================================
# ROW 3:
# POSITIVE WEIGHT STRENGTH
# ============================================================

for col, layer in enumerate(LAYERS):

    ax = axes[2, col]

    sub = df[
        df["layer"] == layer
    ]

    plot_metric(
        ax=ax,
        data=sub,
        metric="mean_positive",
        groups=GROUPS,
        ylabel=(
            "Mean positive weight"
            if col == 0
            else None
        ),
    )


# ============================================================
# ROW 4:
# NEGATIVE WEIGHT STRENGTH
# ============================================================

for col, layer in enumerate(LAYERS):

    ax = axes[3, col]

    sub = df[
        df["layer"] == layer
    ]

    plot_metric(
        ax=ax,
        data=sub,
        metric="mean_negative",
        groups=GROUPS,
        ylabel=(
            "Mean negative weight"
            if col == 0
            else None
        ),
    )


# ============================================================
# LEGENDS
# ============================================================

# Remove all legends first
for ax in axes.ravel():

    legend = ax.get_legend()

    if legend is not None:
        legend.remove()


# Total weight legend
axes[0, 0].legend(
    frameon=False,
    fontsize=8,
)


# Main three-group legend
axes[1, 0].legend(
    frameon=False,
    fontsize=8,
)


# ============================================================
# FINAL LAYOUT
# ============================================================

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

svg_path = (
    OUT
    / "weights_vs_dendrites_dANN_dANNC.svg"
)

pdf_path = (
    OUT
    / "weights_vs_dendrites_dANN_dANNC.pdf"
)

csv_path = (
    OUT
    / "weights_vs_dendrites_dANN_dANNC.csv"
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

df.to_csv(
    csv_path,
    index=False,
)

plt.close(fig)


print("\nSaved:")
print(svg_path)
print(pdf_path)
print(csv_path)