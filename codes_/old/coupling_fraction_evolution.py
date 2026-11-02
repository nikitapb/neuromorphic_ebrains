# coupling_fraction_weight_evolution.py

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras

# Needed to reload dANN-C models
from opt import DendriticCoupling
from plotting_functions import my_style


# ============================================================
# SETTINGS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

BASE = (
    ROOT
    / "DATA_C"
    / "results_fmnist_1_layer"
)

OUT = ROOT / "FinalFigs_manuscript"
OUT.mkdir(exist_ok=True)


# Keep architecture FIXED.
D = 8
S = 512

SIGMA = 0.0

TRIALS = [1, 2, 3, 4, 5]


# ============================================================
# CONDITIONS
# ============================================================

conditions = {

    20: (
        BASE
        / "dend_ann_local_rfs_C"
    ),

    30: (
        BASE
        / "dend_ann_local_rfs_C_frac30_lam0p1"
    ),

    40: (
        BASE
        / "dend_ann_local_rfs_C_frac40_lam0p1"
    ),
}


# ============================================================
# LAYERS
# ============================================================

layers = [
    "Input → dendrite",
    "Dendrite → soma",
    "Soma → output",
]


# ============================================================
# EXTRACT WEIGHTS
# ============================================================

rows = []


def add_weights(
    weights,
    trial,
    fraction,
    layer,
    group,
):

    weights = np.asarray(
        weights
    ).ravel()

    # Remove masked/non-existent connections
    weights = weights[
        weights != 0
    ]

    for w in weights:

        rows.append({

            "trial": trial,
            "fraction": fraction,
            "layer": layer,
            "group": group,
            "weight": float(w),

        })


for fraction, folder in conditions.items():

    n_coupled = int(
        round(
            S * fraction / 100
        )
    )

    print("\n========================================")
    print(
        f"{fraction}% coupled "
        f"({n_coupled}/{S} somata)"
    )
    print("========================================")


    for trial in TRIALS:

        fname = folder / (

            f"model_sigma_{SIGMA}"
            f"_trial_{trial}"
            f"_dends_{D}"
            f"_soma_{S}.keras"

        )


        if not fname.exists():

            raise FileNotFoundError(
                f"\nMissing model:\n{fname}"
            )


        print(
            f"Loading trial {trial}: "
            f"{fname.name}"
        )


        model = keras.models.load_model(
            fname
        )


        # ----------------------------------------------------
        # Layer 1:
        # INPUT -> DENDRITE
        #
        # columns correspond to dendrites.
        # D consecutive dendrites belong to one soma.
        # ----------------------------------------------------

        W_input = (
            model
            .get_layer("dend_1")
            .get_weights()[0]
        )


        for s in range(S):

            group = (
                "Coupled"
                if s < n_coupled
                else "Uncoupled"
            )

            d0 = s * D
            d1 = (s + 1) * D

            add_weights(

                W_input[:, d0:d1],

                trial,
                fraction,

                "Input → dendrite",

                group,
            )


        # ----------------------------------------------------
        # Layer 2:
        # DENDRITE -> SOMA
        #
        # Only real dendrite -> own soma connections.
        # ----------------------------------------------------

        W_soma = (
            model
            .get_layer("soma_1")
            .get_weights()[0]
        )


        for s in range(S):

            group = (
                "Coupled"
                if s < n_coupled
                else "Uncoupled"
            )

            d0 = s * D
            d1 = (s + 1) * D

            add_weights(

                W_soma[d0:d1, s],

                trial,
                fraction,

                "Dendrite → soma",

                group,
            )


        # ----------------------------------------------------
        # Layer 3:
        # SOMA -> OUTPUT
        # ----------------------------------------------------

        W_output = (
            model
            .get_layer("output")
            .get_weights()[0]
        )


        for s in range(S):

            group = (
                "Coupled"
                if s < n_coupled
                else "Uncoupled"
            )


            add_weights(

                W_output[s, :],

                trial,
                fraction,

                "Soma → output",

                group,
            )


        keras.backend.clear_session()


# ============================================================
# RAW DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows
)


print("\nWeights extracted:")

print(
    df.groupby(
        [
            "fraction",
            "layer",
            "group",
        ]
    ).size()
)


# ============================================================
# TRIAL-LEVEL SUMMARY
#
# IMPORTANT:
# Statistics are performed on trials,
# NOT individual weights.
# ============================================================

summary_rows = []


for (
    trial,
    fraction,
    layer,
    group,
), g in df.groupby(
    [
        "trial",
        "fraction",
        "layer",
        "group",
    ]
):

    w = g[
        "weight"
    ].to_numpy()


    positive = w[
        w > 0
    ]

    negative = w[
        w < 0
    ]


    summary_rows.append({

        "trial":
            trial,

        "fraction":
            fraction,

        "layer":
            layer,

        "group":
            group,

        "n_weights":
            len(w),

        "fraction_positive":
            np.mean(
                w > 0
            ),

        "fraction_negative":
            np.mean(
                w < 0
            ),

        "mean_positive":
            np.mean(
                positive
            )
            if len(positive)
            else np.nan,

        "mean_negative":
            np.mean(
                negative
            )
            if len(negative)
            else np.nan,

        "mean_signed":
            np.mean(w),

    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========================================")
print("TRIAL-LEVEL SUMMARY")
print("========================================")


print(

    summary.groupby(
        [
            "fraction",
            "layer",
            "group",
        ]
    )[
        [
            "fraction_positive",
            "mean_positive",
            "mean_negative",
            "mean_signed",
        ]
    ]

    .agg(
        [
            "mean",
            "std",
        ]
    )

    .round(4)
)


# ============================================================
# STATISTICS
#
# Friedman:
# repeated-measures test across 20 / 30 / 40%
#
# Trials are matched because trial 1 is generated from the
# same seed across conditions, etc.
# ============================================================

metrics = [
    "fraction_positive",
    "fraction_negative",
    "mean_positive",
    "mean_negative",
]


stats_rows = []


for layer in layers:

    for group in [
        "Coupled",
        "Uncoupled",
    ]:

        for metric in metrics:

            d = (

                summary[
                    (
                        summary["layer"]
                        == layer
                    )
                    &
                    (
                        summary["group"]
                        == group
                    )
                ]

                .pivot(
                    index="trial",
                    columns="fraction",
                    values=metric,
                )

                .dropna()
            )


            if not all(
                f in d.columns
                for f in [20, 30, 40]
            ):
                continue


            stat, p = friedmanchisquare(
                d[20],
                d[30],
                d[40],
            )


            stats_rows.append({

                "layer":
                    layer,

                "group":
                    group,

                "metric":
                    metric,

                "test":
                    "Friedman",

                "statistic":
                    stat,

                "p":
                    p,

            })


stats = pd.DataFrame(
    stats_rows
)


print("\n========================================")
print("FRIEDMAN TESTS")
print("========================================")

print(
    stats.round(5)
)


# ============================================================
# OPTIONAL PAIRED 20 vs 40 COMPARISON
#
# This gives a simple effect comparison spanning the complete
# fraction manipulation.
# ============================================================

pair_rows = []


for layer in layers:

    for group in [
        "Coupled",
        "Uncoupled",
    ]:

        for metric in metrics:

            d = (

                summary[
                    (
                        summary["layer"]
                        == layer
                    )
                    &
                    (
                        summary["group"]
                        == group
                    )
                ]

                .pivot(
                    index="trial",
                    columns="fraction",
                    values=metric,
                )

                .dropna()
            )


            if (
                20 not in d.columns
                or
                40 not in d.columns
            ):
                continue


            stat, p = wilcoxon(
                d[20],
                d[40],
            )


            pair_rows.append({

                "layer":
                    layer,

                "group":
                    group,

                "metric":
                    metric,

                "comparison":
                    "20 vs 40",

                "W":
                    stat,

                "p":
                    p,

            })


pair_stats = pd.DataFrame(
    pair_rows
)


print("\n========================================")
print("PAIRED 20% vs 40%")
print("========================================")

print(
    pair_stats.round(5)
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


colors = {

    "Uncoupled":
        "#969696",

    "Coupled":
        "#8c2d8c",
}


# ============================================================
# FIGURE
#
# rows:
#   layer
#
# columns:
#   fraction positive
#   fraction negative
#   mean positive
#   mean negative
# ============================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(12.0, 8.0),
)


metric_info = {

    "fraction_positive":
        "Fraction positive weights",

    "fraction_negative":
        "Fraction negative weights",

    "mean_positive":
        "Mean positive weight",

    "mean_negative":
        "Mean negative weight",
}


for row, layer in enumerate(layers):

    ds = summary[
        summary["layer"] == layer
    ]


    for col, (metric, ylabel) in enumerate(
        metric_info.items()
    ):

        ax = axes[row, col]


        # ====================================================
        # INDIVIDUAL TRIAL TRAJECTORIES
        # ====================================================

        for group in [
            "Uncoupled",
            "Coupled",
        ]:

            dg = ds[
                ds["group"] == group
            ]


            for trial in TRIALS:

                dt = (
                    dg[
                        dg["trial"] == trial
                    ]
                    .sort_values("fraction")
                )


                ax.plot(
                    dt["fraction"],
                    dt[metric],
                    color=colors[group],
                    alpha=0.20,
                    linewidth=0.8,
                )


            # ================================================
            # MEAN ± SEM
            # ================================================

            means = (
                dg.groupby("fraction")[metric]
                .mean()
            )

            sems = (
                dg.groupby("fraction")[metric]
                .sem()
            )


            x = means.index.to_numpy()
            y = means.to_numpy()
            e = sems.to_numpy()


            ax.errorbar(
                x,
                y,
                yerr=e,
                marker="o",
                markersize=6,
                linewidth=2,
                capsize=3,
                color=colors[group],
                label=group,
            )


        # ====================================================
        # FORMAT
        # ====================================================

        ax.set_xticks(
            [20, 30, 40]
        )


        if row == 2:

            ax.set_xlabel(
                "Coupled somata (%)"
            )

        else:

            ax.set_xlabel("")


        ax.set_ylabel(
            ylabel
            if col == 0
            else ""
        )


        if row == 0:

            ax.set_title(
                ylabel
            )


        if col == 0:

            ax.text(
                -0.38,
                0.5,
                layer,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontweight="bold",
            )


        if (
            row == 0
            and col == 3
        ):

            ax.legend(
                frameon=False,
            )


# ============================================================
# FRACTION PANELS
#
# Both fractions naturally range from 0 to 1.
# ============================================================

for row in range(3):

    axes[row, 0].set_ylim(
        0,
        1,
    )

    axes[row, 1].set_ylim(
        0,
        1,
    )


sns.despine()

fig.tight_layout()


# Fraction-positive is naturally bounded
for row in range(3):

    axes[
        row,
        0
    ].set_ylim(
        0,
        1,
    )


sns.despine()

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

pdf = (
    OUT
    / "dANNC_weight_evolution_coupling_fraction.pdf"
)

svg = (
    OUT
    / "dANNC_weight_evolution_coupling_fraction.svg"
)

raw_csv = (
    OUT
    / "dANNC_weight_evolution_coupling_fraction_raw.csv"
)

summary_csv = (
    OUT
    / "dANNC_weight_evolution_coupling_fraction_summary.csv"
)

stats_csv = (
    OUT
    / "dANNC_weight_evolution_coupling_fraction_stats.csv"
)

pair_stats_csv = (
    OUT
    / "dANNC_weight_evolution_coupling_fraction_20_vs_40.csv"
)


fig.savefig(
    pdf,
    bbox_inches="tight",
)

fig.savefig(
    svg,
    bbox_inches="tight",
    dpi=600,
)

plt.close(fig)


df.to_csv(
    raw_csv,
    index=False,
)

summary.to_csv(
    summary_csv,
    index=False,
)

stats.to_csv(
    stats_csv,
    index=False,
)

pair_stats.to_csv(
    pair_stats_csv,
    index=False,
)


print("\n========================================")
print("SAVED")
print("========================================")

print(pdf)
print(svg)
print(raw_csv)
print(summary_csv)
print(stats_csv)
print(pair_stats_csv)