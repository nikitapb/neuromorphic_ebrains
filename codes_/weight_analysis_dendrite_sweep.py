# weight_analysis_dendrite_sweep.py

import pathlib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras

from opt import DendriticCoupling
from plotting_functions import my_style


# ============================================================
# PATHS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

DATA_C = (
    ROOT
    / "DATA_C"
    / "results_fmnist_1_layer"
)

OUT = (
    ROOT
    / "FinalFigs_manuscript"
)

OUT.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

S = 512

DENDS = [
    2,
    4,
    8,
    16,
    32,
    64,
]

TRIALS = [
    1,
    2,
    3,
    4,
    5,
]

SIGMA = 0.0


# ============================================================
# MODELS
#
# This analysis deliberately contains only our dANN-C models.
#
# dANN-LRF is NOT included because we do not currently have
# the complete S=512 D=2..64 model series locally.
# ============================================================

MODELS = {

    "dANN-C 10%, λ=0.10": {
        "fraction": 10,
        "lambda": 0.10,
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac10_lam0p1",
    },

    "dANN-C 20%, λ=0.10": {
        "fraction": 20,
        "lambda": 0.10,
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C",
    },

    "dANN-C 30%, λ=0.10": {
        "fraction": 30,
        "lambda": 0.10,
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac30_lam0p1",
    },

    "dANN-C 40%, λ=0.10": {
        "fraction": 40,
        "lambda": 0.10,
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac40_lam0p1",
    },

    "dANN-C 20%, λ=0.05": {
        "fraction": 20,
        "lambda": 0.05,
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac20_lam0p05",
    },

    "dANN-C 20%, λ=0.20": {
        "fraction": 20,
        "lambda": 0.20,
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac20_lam0p2",
    },
}


COUPLING_MODELS = [

    "dANN-C 10%, λ=0.10",

    "dANN-C 20%, λ=0.10",

    "dANN-C 30%, λ=0.10",

    "dANN-C 40%, λ=0.10",
]


LAMBDA_MODELS = [

    "dANN-C 20%, λ=0.05",

    "dANN-C 20%, λ=0.10",

    "dANN-C 20%, λ=0.20",
]


COLORS = {

    "dANN-C 10%, λ=0.10":
        "#d8b4d8",

    "dANN-C 20%, λ=0.10":
        "#8c2d8c",

    "dANN-C 30%, λ=0.10":
        "#b2182b",

    "dANN-C 40%, λ=0.10":
        "#67001f",

    "dANN-C 20%, λ=0.05":
        "#67a9cf",

    "dANN-C 20%, λ=0.20":
        "#2166ac",
}


LAYERS = [
    "Input → dendrite",
    "Dendrite → soma",
    "Soma → output",
]


# ============================================================
# STYLE
# ============================================================

sns.set_style("white")

plt.rcParams.update(
    my_style()
)


# ============================================================
# HELPERS
# ============================================================

def remove_zeros(weights):

    weights = np.asarray(
        weights,
        dtype=float,
    ).ravel()

    return weights[
        weights != 0
    ]



def find_model_file(
    model_name,
    trial,
    D,
):

    folder = (
        MODELS[
            model_name
        ]["folder"]
    )


    fname = folder / (

        f"model_sigma_{SIGMA}"
        f"_trial_{trial}"
        f"_dends_{D}"
        f"_soma_{S}.keras"
    )


    if not fname.exists():

        raise FileNotFoundError(
            str(fname)
        )


    return fname



def load_weights(
    model_name,
    trial,
    D,
):

    fname = find_model_file(

        model_name,
        trial,
        D,
    )


    model = keras.models.load_model(
        fname
    )


    # --------------------------------------------------------
    # Input -> dendrite
    # --------------------------------------------------------

    W_input = (
        model
        .get_layer("dend_1")
        .get_weights()[0]
    )


    w_input = remove_zeros(
        W_input
    )


    # --------------------------------------------------------
    # Dendrite -> soma
    # --------------------------------------------------------

    W_soma = (
        model
        .get_layer("soma_1")
        .get_weights()[0]
    )


    soma_weights = []


    for soma_idx in range(
        S
    ):

        d0 = (
            soma_idx
            * D
        )

        d1 = (
            (soma_idx + 1)
            * D
        )


        soma_weights.append(

            W_soma[
                d0:d1,
                soma_idx
            ]
        )


    w_soma = remove_zeros(

        np.concatenate(
            soma_weights
        )
    )


    # --------------------------------------------------------
    # Soma -> output
    # --------------------------------------------------------

    W_output = (
        model
        .get_layer("output")
        .get_weights()[0]
    )


    w_output = remove_zeros(
        W_output
    )


    keras.backend.clear_session()


    return {

        "Input → dendrite":
            w_input,

        "Dendrite → soma":
            w_soma,

        "Soma → output":
            w_output,
    }



def summarize_weights(
    weights,
):

    weights = remove_zeros(
        weights
    )


    positive = weights[
        weights > 0
    ]


    negative = weights[
        weights < 0
    ]


    return {

        "fraction_positive":
            np.mean(
                weights > 0
            ),

        "fraction_negative":
            np.mean(
                weights < 0
            ),

        "mean_positive":
            (
                np.mean(positive)
                if len(positive)
                else np.nan
            ),

        "mean_negative":
            (
                np.mean(negative)
                if len(negative)
                else np.nan
            ),

        "mean_signed":
            np.mean(
                weights
            ),
    }


# ============================================================
# AVAILABILITY CHECK
# ============================================================

print("\n==========================================")
print("DENDRITE-SWEEP MODEL AVAILABILITY")
print("==========================================")


for model_name in MODELS:

    print(
        f"\n{model_name}"
    )


    total = 0


    for D in DENDS:

        n = 0


        for trial in TRIALS:

            try:

                find_model_file(
                    model_name,
                    trial,
                    D,
                )

                n += 1
                total += 1

            except FileNotFoundError:

                pass


        print(
            f"  D={D:2d}: "
            f"{n}/5"
        )


    print(
        f"  TOTAL: "
        f"{total}/30"
    )


    if total != 30:

        raise RuntimeError(

            f"\n{model_name} does not have the complete "
            f"S={S} dendrite sweep."
        )


# ============================================================
# EXTRACT TRIAL-LEVEL SUMMARY
#
# This stays small:
# model × D × trial × layer
# ============================================================

summary_rows = []


print("\n==========================================")
print("EXTRACTING D-SWEEP WEIGHTS")
print("==========================================")


for model_name in MODELS:

    print(
        f"\n{model_name}"
    )


    for D in DENDS:

        print(
            f"  D={D}"
        )


        for trial in TRIALS:

            weights = load_weights(

                model_name,
                trial,
                D,
            )


            for layer in LAYERS:

                stats = summarize_weights(

                    weights[
                        layer
                    ]
                )


                row = {

                    "model":
                        model_name,

                    "trial":
                        trial,

                    "D":
                        D,

                    "S":
                        S,

                    "fraction":
                        MODELS[
                            model_name
                        ]["fraction"],

                    "lambda":
                        MODELS[
                            model_name
                        ]["lambda"],

                    "layer":
                        layer,
                }


                row.update(
                    stats
                )


                summary_rows.append(
                    row
                )


SUMMARY = pd.DataFrame(
    summary_rows
)


SUMMARY_FILE = (

    OUT
    / "weights_dendrite_sweep_S512_summary.csv"
)


SUMMARY.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# LINE-PLOT HELPER
#
# Thin lines:
# individual training trials
#
# Thick line:
# mean across trials
#
# Error:
# SEM
# ============================================================

def plot_D_metric(
    ax,
    data,
    models,
    metric,
):

    for model_name in models:

        d = data[
            data[
                "model"
            ] == model_name
        ]


        # ----------------------------------------------------
        # Individual trials
        # ----------------------------------------------------

        for trial in TRIALS:

            dt = (

                d[
                    d[
                        "trial"
                    ] == trial
                ]

                .sort_values(
                    "D"
                )
            )


            ax.plot(

                dt[
                    "D"
                ],

                dt[
                    metric
                ],

                color=
                    COLORS[
                        model_name
                    ],

                alpha=
                    0.15,

                linewidth=
                    0.8,
            )


        # ----------------------------------------------------
        # Mean ± SEM
        # ----------------------------------------------------

        means = (

            d.groupby(
                "D"
            )[
                metric
            ]

            .mean()

            .reindex(
                DENDS
            )
        )


        sems = (

            d.groupby(
                "D"
            )[
                metric
            ]

            .sem()

            .reindex(
                DENDS
            )
        )


        ax.errorbar(

            DENDS,

            means,

            yerr=
                sems,

            color=
                COLORS[
                    model_name
                ],

            marker=
                "o",

            markersize=
                5,

            linewidth=
                1.8,

            capsize=
                2,

            label=
                model_name,
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


# ============================================================
# FIGURE 1
#
# POSITIVE / NEGATIVE FRACTION VS D
# ============================================================

def make_fraction_D_figure(
    models,
    filename,
):

    fig, axes = plt.subplots(

        3,
        2,

        figsize=(
            8,
            8,
        ),

        sharex=True,
    )


    for row, layer in enumerate(
        LAYERS
    ):

        d = SUMMARY[

            (
                SUMMARY[
                    "model"
                ].isin(
                    models
                )
            )

            &

            (
                SUMMARY[
                    "layer"
                ] == layer
            )

        ]


        for col, (
            metric,
            title
        ) in enumerate(

            [
                (
                    "fraction_positive",
                    "Fraction positive",
                ),

                (
                    "fraction_negative",
                    "Fraction negative",
                ),
            ]
        ):

            ax = axes[
                row,
                col
            ]


            plot_D_metric(

                ax,
                d,
                models,
                metric,
            )


            if row == 0:

                ax.set_title(
                    title
                )


            if row == 2:

                ax.set_xlabel(
                    "Dendrites per soma"
                )


            if col == 0:

                ax.set_ylabel(
                    layer
                )


    axes[
        0,
        1
    ].legend(

        frameon=False,

        fontsize=7,

        bbox_to_anchor=(
            1.02,
            1,
        ),
    )


    sns.despine()

    fig.tight_layout()


    fig.savefig(

        OUT
        / filename,

        bbox_inches="tight",
    )


    plt.close(fig)


make_fraction_D_figure(

    COUPLING_MODELS,

    "weights_D_sweep_fraction_coupling.pdf",
)


make_fraction_D_figure(

    LAMBDA_MODELS,

    "weights_D_sweep_fraction_lambda.pdf",
)


# ============================================================
# FIGURE 2
#
# POSITIVE / NEGATIVE STRENGTH VS D
# ============================================================

def make_strength_D_figure(
    models,
    filename,
):

    fig, axes = plt.subplots(

        3,
        2,

        figsize=(
            8,
            8,
        ),

        sharex=True,
    )


    for row, layer in enumerate(
        LAYERS
    ):

        d = SUMMARY[

            (
                SUMMARY[
                    "model"
                ].isin(
                    models
                )
            )

            &

            (
                SUMMARY[
                    "layer"
                ] == layer
            )

        ]


        for col, (
            metric,
            title
        ) in enumerate(

            [
                (
                    "mean_positive",
                    "Positive weight strength",
                ),

                (
                    "mean_negative",
                    "Negative weight strength",
                ),
            ]
        ):

            ax = axes[
                row,
                col
            ]


            plot_D_metric(

                ax,
                d,
                models,
                metric,
            )


            if row == 0:

                ax.set_title(
                    title
                )


            if row == 2:

                ax.set_xlabel(
                    "Dendrites per soma"
                )


            if col == 0:

                ax.set_ylabel(
                    layer
                )


    axes[
        0,
        1
    ].legend(

        frameon=False,

        fontsize=7,

        bbox_to_anchor=(
            1.02,
            1,
        ),
    )


    sns.despine()

    fig.tight_layout()


    fig.savefig(

        OUT
        / filename,

        bbox_inches="tight",
    )


    plt.close(fig)


make_strength_D_figure(

    COUPLING_MODELS,

    "weights_D_sweep_strength_coupling.pdf",
)


make_strength_D_figure(

    LAMBDA_MODELS,

    "weights_D_sweep_strength_lambda.pdf",
)


# ============================================================
# FULL DISTRIBUTION FIGURES
#
# These are generated directly from the models rather than
# storing millions of individual weights in a dataframe.
# ============================================================

def make_distribution_D_figure(
    models,
    filename,
):

    fig, axes = plt.subplots(

        3,
        len(DENDS),

        figsize=(
            15,
            7.5,
        ),

        sharey="row",
    )


    for col, D in enumerate(
        DENDS
    ):

        print(
            f"\nBuilding distribution figure: D={D}"
        )


        # ----------------------------------------------------
        # Pool five trials for each model/layer at this D.
        # ----------------------------------------------------

        pooled = {

            model_name: {
                layer: []
                for layer in LAYERS
            }

            for model_name in models
        }


        for model_name in models:

            for trial in TRIALS:

                weights = load_weights(

                    model_name,
                    trial,
                    D,
                )


                for layer in LAYERS:

                    pooled[
                        model_name
                    ][
                        layer
                    ].append(

                        weights[
                            layer
                        ]
                    )


            for layer in LAYERS:

                pooled[
                    model_name
                ][
                    layer
                ] = np.concatenate(

                    pooled[
                        model_name
                    ][
                        layer
                    ]
                )


        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        for row, layer in enumerate(
            LAYERS
        ):

            ax = axes[
                row,
                col
            ]


            for model_name in models:

                sns.kdeplot(

                    x=
                        pooled[
                            model_name
                        ][
                            layer
                        ],

                    ax=
                        ax,

                    color=
                        COLORS[
                            model_name
                        ],

                    linewidth=
                        1.5,

                    fill=
                        False,

                    common_norm=
                        False,

                    label=
                        (
                            model_name
                            if col == 0
                            else None
                        ),
                )


            ax.axvline(

                0,

                color="0.7",

                ls="--",

                lw=0.8,
            )


            if row == 0:

                ax.set_title(
                    f"D = {D}"
                )


            if col == 0:

                ax.set_ylabel(
                    f"{layer}\nProbability density"
                )

            else:

                ax.set_ylabel("")


            if row == 2:

                ax.set_xlabel(
                    "Weight"
                )

            else:

                ax.set_xlabel("")


    axes[
        0,
        0
    ].legend(

        frameon=False,

        fontsize=7,
    )


    sns.despine()

    fig.tight_layout()


    fig.savefig(

        OUT
        / filename,

        bbox_inches="tight",
    )


    plt.close(fig)


# ============================================================
# DISTRIBUTION EVOLUTION — COUPLING FRACTION
# ============================================================

make_distribution_D_figure(

    COUPLING_MODELS,

    "weights_D_sweep_distributions_coupling.pdf",
)


# ============================================================
# DISTRIBUTION EVOLUTION — LAMBDA
# ============================================================

make_distribution_D_figure(

    LAMBDA_MODELS,

    "weights_D_sweep_distributions_lambda.pdf",
)


# ============================================================
# DONE
# ============================================================

print("\n==========================================")
print("DENDRITE-SWEEP WEIGHT ANALYSIS COMPLETE")
print("==========================================")


print(
    f"\nS = {S}"
)

print(
    f"D = {DENDS}"
)


print("\nSaved:")


for fname in [

    "weights_dendrite_sweep_S512_summary.csv",

    "weights_D_sweep_fraction_coupling.pdf",

    "weights_D_sweep_fraction_lambda.pdf",

    "weights_D_sweep_strength_coupling.pdf",

    "weights_D_sweep_strength_lambda.pdf",

    "weights_D_sweep_distributions_coupling.pdf",

    "weights_D_sweep_distributions_lambda.pdf",
]:

    print(
        OUT
        / fname
    )