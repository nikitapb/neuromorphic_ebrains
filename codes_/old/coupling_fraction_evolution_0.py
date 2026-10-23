import pathlib
import pickle

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

from plotting_functions import (
    my_style,
    fix_names,
    find_best_models,
)


# ============================================================
# SETTINGS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

BASE = (
    ROOT
    / "DATA_C"
    / "results_fmnist_1_layer"
)

ORIGINAL_DATA = (
    ROOT
    / "DATA"
    / "results_fmnist_1_layer"
)

WEIGHTS_FILE = (
    ROOT
    / "DATA"
    / "weights_best_models.pkl"
)

OUT = (
    ROOT
    / "FinalFigs_manuscript"
)

OUT.mkdir(
    exist_ok=True
)


# Keep architecture FIXED for dANN-C
D = 8
D = 4
S = 512

SIGMA = 0.0

TRIALS = [
    1,
    2,
    3,
    4,
    5,
]


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
# CHECK WHICH dANN-LRF ARCHITECTURE IS IN
# weights_best_models.pkl
# ============================================================

print("\n========================================")
print("CHECKING ORIGINAL dANN-LRF")
print("========================================")


with open(
    ORIGINAL_DATA
    / "output_all_final.pkl",
    "rb",
) as f:

    original_results = pickle.load(f)


df_original_test = fix_names(
    original_results["testing"]
).copy()


df_original_test[
    "test_acc"
] *= 100


models_best = find_best_models(

    df_original_test,

    [
        "dANN-LRF",
        "vANN",
    ],

    metric="accuracy",
)


dann_best_D = int(
    models_best[
        "dANN-LRF"
    ][0]
)

dann_best_S = int(
    models_best[
        "dANN-LRF"
    ][1]
)


print(
    f"dANN-LRF stored in weights_best_models.pkl:"
)

print(
    f"  D = {dann_best_D}"
)

print(
    f"  S = {dann_best_S}"
)

print(
    f"\ndANN-C analysis:"
)

print(
    f"  D = {D}"
)

print(
    f"  S = {S}"
)


if (
    dann_best_D != D
    or
    dann_best_S != S
):

    print(
        "\nWARNING:"
    )

    print(
        "The dANN-LRF weights in weights_best_models.pkl "
        "do NOT correspond to D=8, S=512."
    )

    print(
        "They will still be plotted as the original "
        "dANN-LRF reference, but they are not an exactly "
        "architecture-matched 0% control."
    )

else:

    print(
        "\nPerfect: dANN-LRF is architecture-matched "
        "to D=8, S=512."
    )


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


    # Remove structural zeros / padding
    weights = weights[
        weights != 0
    ]


    for w in weights:

        rows.append(
            {
                "trial":
                    trial,

                "fraction":
                    fraction,

                "layer":
                    layer,

                "group":
                    group,

                "weight":
                    float(w),
            }
        )


# ============================================================
# 0% COUPLING:
# ORIGINAL dANN-LRF
#
# weights_best_models.pkl stores:
#
#     rows = weights
#     columns = trials 1-5
#
# for each layer.
# ============================================================

print("\n========================================")
print("LOADING dANN-LRF AS 0% COUPLING")
print("========================================")


with open(
    WEIGHTS_FILE,
    "rb",
) as f:

    original_weights = pickle.load(f)


W_dann_input = np.asarray(
    original_weights[
        "dendrites"
    ][
        "dend_ann_local_rfs"
    ]
)


W_dann_soma = np.asarray(
    original_weights[
        "soma"
    ][
        "dend_ann_local_rfs"
    ]
)


W_dann_output = np.asarray(
    original_weights[
        "output"
    ][
        "dend_ann_local_rfs"
    ]
)


print(
    "Input → dendrite:",
    W_dann_input.shape,
)

print(
    "Dendrite → soma:",
    W_dann_soma.shape,
)

print(
    "Soma → output:",
    W_dann_output.shape,
)


# Confirm that trial axis is present
if W_dann_input.shape[1] != len(TRIALS):

    raise ValueError(
        "Unexpected number of dANN-LRF trials in "
        "weights_best_models.pkl"
    )


for trial_idx, trial in enumerate(
    TRIALS
):

    print(
        f"Adding dANN-LRF trial {trial}"
    )


    add_weights(

        W_dann_input[
            :,
            trial_idx,
        ],

        trial=trial,

        fraction=0,

        layer="Input → dendrite",

        group="dANN-LRF",
    )


    add_weights(

        W_dann_soma[
            :,
            trial_idx,
        ],

        trial=trial,

        fraction=0,

        layer="Dendrite → soma",

        group="dANN-LRF",
    )


    add_weights(

        W_dann_output[
            :,
            trial_idx,
        ],

        trial=trial,

        fraction=0,

        layer="Soma → output",

        group="dANN-LRF",
    )


# ============================================================
# dANN-C CONDITIONS
# ============================================================

for fraction, folder in conditions.items():

    n_coupled = int(
        round(
            S
            * fraction
            / 100
        )
    )


    print(
        "\n========================================"
    )

    print(
        f"{fraction}% coupled "
        f"({n_coupled}/{S} somata)"
    )

    print(
        "========================================"
    )


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
        # ----------------------------------------------------

        W_input = (

            model
            .get_layer(
                "dend_1"
            )
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

                W_input[
                    :,
                    d0:d1,
                ],

                trial,

                fraction,

                "Input → dendrite",

                group,
            )


        # ----------------------------------------------------
        # Layer 2:
        # DENDRITE -> SOMA
        # ----------------------------------------------------

        W_soma = (

            model
            .get_layer(
                "soma_1"
            )
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

                W_soma[
                    d0:d1,
                    s,
                ],

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
            .get_layer(
                "output"
            )
            .get_weights()[0]

        )


        for s in range(S):

            group = (

                "Coupled"

                if s < n_coupled

                else "Uncoupled"

            )


            add_weights(

                W_output[
                    s,
                    :,
                ],

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


print(
    "\nWeights extracted:"
)


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


    summary_rows.append(
        {

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
                (
                    np.mean(
                        positive
                    )
                    if len(
                        positive
                    )
                    else np.nan
                ),

            "mean_negative":
                (
                    np.mean(
                        negative
                    )
                    if len(
                        negative
                    )
                    else np.nan
                ),

            "mean_signed":
                np.mean(
                    w
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "TRIAL-LEVEL SUMMARY"
)

print(
    "========================================"
)


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
            "fraction_negative",
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

    .round(
        4
    )
)


# ============================================================
# STATISTICS
#
# Keep statistics only within dANN-C:
# 20 / 30 / 40 %
#
# dANN-LRF is the 0% reference point, but is NOT included
# in these Friedman tests because it is not split into
# coupled / uncoupled populations.
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
                        summary[
                            "layer"
                        ]
                        == layer
                    )
                    &
                    (
                        summary[
                            "group"
                        ]
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
                for f in [
                    20,
                    30,
                    40,
                ]
            ):

                continue


            stat, p = friedmanchisquare(

                d[20],

                d[30],

                d[40],
            )


            stats_rows.append(
                {

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
                }
            )


stats = pd.DataFrame(
    stats_rows
)


print(
    "\n========================================"
)

print(
    "FRIEDMAN TESTS"
)

print(
    "========================================"
)


print(
    stats.round(
        5
    )
)


# ============================================================
# PAIRED 20 vs 40
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
                        summary[
                            "layer"
                        ]
                        == layer
                    )
                    &
                    (
                        summary[
                            "group"
                        ]
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


            pair_rows.append(
                {

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
                }
            )


pair_stats = pd.DataFrame(
    pair_rows
)


print(
    "\n========================================"
)

print(
    "PAIRED 20% vs 40%"
)

print(
    "========================================"
)


print(
    pair_stats.round(
        5
    )
)


# ============================================================
# STYLE
# ============================================================

sns.set_style(
    "white"
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


colors = {

    "dANN-LRF":
        "#000000",

    "Uncoupled":
        "#969696",

    "Coupled":
        "#8c2d8c",
}


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(

    3,

    4,

    figsize=(
        12.0,
        8.0,
    ),
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


for row, layer in enumerate(
    layers
):

    ds = summary[
        summary[
            "layer"
        ]
        == layer
    ]


    for col, (
        metric,
        ylabel,
    ) in enumerate(
        metric_info.items()
    ):

        ax = axes[
            row,
            col,
        ]


        # ====================================================
        # dANN-LRF CONTROL:
        # 0% coupling
        # ====================================================

        dctrl = ds[
            ds[
                "group"
            ]
            == "dANN-LRF"
        ]


        if not dctrl.empty:

            ctrl_vals = (
                dctrl[
                    metric
                ]
                .dropna()
            )


            # Individual trials
            ax.scatter(

                np.zeros(
                    len(
                        ctrl_vals
                    )
                ),

                ctrl_vals,

                color=colors[
                    "dANN-LRF"
                ],

                s=18,

                alpha=0.25,

                zorder=3,
            )


            ctrl_mean = (
                ctrl_vals.mean()
            )


            ctrl_sem = (
                ctrl_vals.sem()
            )


            ax.errorbar(

                0,

                ctrl_mean,

                yerr=ctrl_sem,

                marker="o",

                markersize=7,

                linewidth=2,

                capsize=3,

                color=colors[
                    "dANN-LRF"
                ],

                label="dANN-LRF",

                zorder=5,
            )


        # ====================================================
        # dANN-C:
        # uncoupled and coupled populations
        # ====================================================

        for group in [
            "Uncoupled",
            "Coupled",
        ]:

            dg = ds[
                ds[
                    "group"
                ]
                == group
            ]


            # ================================================
            # INDIVIDUAL TRIAL TRAJECTORIES
            # ================================================

            for trial in TRIALS:

                dt = (

                    dg[
                        dg[
                            "trial"
                        ]
                        == trial
                    ]

                    .sort_values(
                        "fraction"
                    )
                )


                ax.plot(

                    dt[
                        "fraction"
                    ],

                    dt[
                        metric
                    ],

                    color=colors[
                        group
                    ],

                    alpha=0.20,

                    linewidth=0.8,
                )


            # ================================================
            # MEAN ± SEM
            # ================================================

            means = (

                dg.groupby(
                    "fraction"
                )[
                    metric
                ]

                .mean()
            )


            sems = (

                dg.groupby(
                    "fraction"
                )[
                    metric
                ]

                .sem()
            )


            x = (
                means.index
                .to_numpy()
            )

            y = (
                means
                .to_numpy()
            )

            e = (
                sems
                .to_numpy()
            )


            ax.errorbar(

                x,

                y,

                yerr=e,

                marker="o",

                markersize=6,

                linewidth=2,

                capsize=3,

                color=colors[
                    group
                ],

                label=group,
            )


        # ====================================================
        # FORMAT
        # ====================================================

        ax.set_xticks(
            [
                0,
                20,
                30,
                40,
            ]
        )


        ax.set_xticklabels(
            [
                "0",
                "20",
                "30",
                "40",
            ]
        )


        if row == 2:

            ax.set_xlabel(
                "Coupled somata (%)"
            )

        else:

            ax.set_xlabel(
                ""
            )


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
            and
            col == 3
        ):

            ax.legend(
                frameon=False
            )


# ============================================================
# FRACTION PANELS
# ============================================================

for row in range(
    3
):

    axes[
        row,
        0
    ].set_ylim(
        0.4,
        0.6,
    )

    axes[
        row,
        1
    ].set_ylim(
        0.4,
        0.6,
    )


sns.despine()

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

pdf = (

    OUT
    / "dANNC_weight_evolution_coupling_fraction_with_dANNLRF.pdf"
)

svg = (

    OUT
    / "dANNC_weight_evolution_coupling_fraction_with_dANNLRF.svg"
)

raw_csv = (

    OUT
    / "dANNC_weight_evolution_coupling_fraction_with_dANNLRF_raw.csv"
)

summary_csv = (

    OUT
    / "dANNC_weight_evolution_coupling_fraction_with_dANNLRF_summary.csv"
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


plt.close(
    fig
)


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


print(
    "\n========================================"
)

print(
    "SAVED"
)

print(
    "========================================"
)

print(
    pdf
)

print(
    svg
)

print(
    raw_csv
)

print(
    summary_csv
)

print(
    stats_csv
)

print(
    pair_stats_csv
)