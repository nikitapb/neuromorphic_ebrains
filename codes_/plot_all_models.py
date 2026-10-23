# plot_all_models.py

import re
import pickle
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import keras

from opt import DendriticCoupling
from plotting_functions import (
    my_style,
    fix_names,
    calculate_best_model,
)


# ============================================================
# PATHS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

DATA = (
    ROOT
    / "DATA"
    / "results_fmnist_1_layer"
)

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

SIGMA = 0.0

N_SYN = 16

N_CLASSES = 10


# ============================================================
# BEST ARCHITECTURE FOR WEIGHT ANALYSIS
# ============================================================

BEST_D = 8
BEST_S = 512


# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODEL_INFO = {

    "dANN-LRF": {
        "folder":
            DATA
            / "dend_ann_local_rfs",

        "fraction":
            0,

        "lambda":
            0.0,
    },


    "dANN-C 10%": {
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac10_lam0p1",

        "fraction":
            10,

        "lambda":
            0.10,
    },


    "dANN-C 20%": {
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C",

        "fraction":
            20,

        "lambda":
            0.10,
    },


    "dANN-C 30%": {
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac30_lam0p1",

        "fraction":
            30,

        "lambda":
            0.10,
    },


    "dANN-C 40%": {
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac40_lam0p1",

        "fraction":
            40,

        "lambda":
            0.10,
    },


    "dANN-C λ=0.05": {
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac20_lam0p05",

        "fraction":
            20,

        "lambda":
            0.05,
    },


    "dANN-C λ=0.20": {
        "folder":
            DATA_C
            / "dend_ann_local_rfs_C_frac20_lam0p2",

        "fraction":
            20,

        "lambda":
            0.20,
    },
}


# ============================================================
# MODEL GROUPS
# ============================================================

ALL_MODELS = [
    "dANN-LRF",
    "dANN-C 10%",
    "dANN-C 20%",
    "dANN-C 30%",
    "dANN-C 40%",
    "dANN-C λ=0.05",
    "dANN-C λ=0.20",
]


COUPLING_MODELS = [
    "dANN-LRF",
    "dANN-C 10%",
    "dANN-C 20%",
    "dANN-C 30%",
    "dANN-C 40%",
]


LAMBDA_MODELS = [
    "dANN-LRF",
    "dANN-C λ=0.05",
    "dANN-C 20%",
    "dANN-C λ=0.20",
]


LAYERS = [
    "Input → dendrite",
    "Dendrite → soma",
    "Soma → output",
]


# ============================================================
# COLORS
# ============================================================

# ------------------------------------------------------------
# dANN-LRF
# ------------------------------------------------------------

COLOR_LRF = "#000000"


# ------------------------------------------------------------
# Coupling-fraction gradients
# ------------------------------------------------------------

COUPLING_COLORS = {

    10:
        "#fee6ce",

    20:
        "#fdae6b",

    30:
        "#e6550d",

    40:
        "#a63603",
}


# ------------------------------------------------------------
# Lambda gradients
# ------------------------------------------------------------

LAMBDA_COLORS = {

    0.05:
        "#bdd7e7",

    0.10:
        "#6baed6",

    0.20:
        "#2171b5",
}


# ------------------------------------------------------------
# Coupled / uncoupled population colors
#
# Used for METRIC evolution.
# ------------------------------------------------------------

GROUP_COLORS = {

    "dANN-LRF":
        "#000000",

    "Uncoupled":
        "#969696",

    "Coupled":
        "#8c2d8c",
}


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
# BASIC HELPERS
# ============================================================

def nparams(
    D,
    S,
    n_syn=N_SYN,
    n_class=N_CLASSES,
):

    return (
        (n_syn + 1)
        * D
        * S

        +

        (D + 1)
        * S

        +

        (S + 1)
        * n_class
    )



def remove_zeros(
    weights,
):

    weights = np.asarray(
        weights,
        dtype=float,
    ).ravel()


    return weights[
        weights != 0
    ]



def extract_scalar(
    x,
):

    return float(
        np.asarray(
            x
        ).squeeze()
    )



def mean_sem(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(values) == 0:

        return (
            np.nan,
            np.nan,
        )


    mean = np.mean(
        values
    )


    if len(values) > 1:

        sem = (

            np.std(
                values,
                ddof=1,
            )

            /

            np.sqrt(
                len(values)
            )
        )

    else:

        sem = np.nan


    return (
        mean,
        sem,
    )



def summarize_weights(
    weights,
):

    weights = remove_zeros(
        weights
    )


    if len(weights) == 0:

        return {

            "fraction_positive":
                np.nan,

            "fraction_negative":
                np.nan,

            "mean_positive":
                np.nan,

            "mean_negative":
                np.nan,
        }


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
                np.mean(
                    positive
                )

                if len(positive)

                else np.nan
            ),

        "mean_negative":
            (
                np.mean(
                    negative
                )

                if len(negative)

                else np.nan
            ),
    }



def zoom_axis(
    ax,
    values,
    margin_fraction=0.20,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(values) == 0:

        return


    ymin = np.min(
        values
    )


    ymax = np.max(
        values
    )


    span = (
        ymax
        - ymin
    )


    if span == 0:

        span = (

            abs(ymin)
            * 0.10

            if ymin != 0

            else 0.1
        )


    margin = (
        span
        * margin_fraction
    )


    ax.set_ylim(

        ymin
        - margin,

        ymax
        + margin,
    )


# ============================================================
# FILE PATTERNS
# ============================================================

RESULT_PATTERN = re.compile(

    r"results_sigma_0\.0"
    r"_trial_(\d+)"
    r"_dends_(\d+)"
    r"_soma_(\d+)\.pkl"
)


MODEL_PATTERN = re.compile(

    r"model_sigma_0\.0"
    r"_trial_(\d+)"
    r"_dends_(\d+)"
    r"_soma_(\d+)\.keras"
)


# ============================================================
# LOAD vANN REFERENCE
# ============================================================

def load_vann_reference():

    fname = (
        DATA
        / "output_all_final.pkl"
    )


    if not fname.exists():

        return None


    with open(
        fname,
        "rb",
    ) as f:

        results = pickle.load(
            f
        )


    df_test = fix_names(
        results[
            "testing"
        ]
    ).copy()


    df_test[
        "test_acc"
    ] *= 100


    vann = df_test[
        df_test[
            "model"
        ] == "vANN"
    ].copy()


    if len(vann) == 0:

        return None


    eval_metrics = (
        calculate_best_model(
            vann
        )
    )


    return {

        "best_loss":
            np.nanmin(
                eval_metrics[
                    :,
                    0
                ]
            ),

        "best_acc":
            np.nanmax(
                eval_metrics[
                    :,
                    1
                ]
            ),

        "best_loss_params":
            eval_metrics[
                np.nanargmin(
                    eval_metrics[
                        :,
                        0
                    ]
                )
            ][
                -1
            ],

        "best_acc_params":
            eval_metrics[
                np.nanargmax(
                    eval_metrics[
                        :,
                        1
                    ]
                )
            ][
                -1
            ],
    }


VANN_REF = (
    load_vann_reference()
)


# ============================================================
# PERFORMANCE DATA
# ============================================================

def load_results_from_folder(
    model_name,
    folder,
    fraction,
    lam,
):

    rows = []


    if not folder.exists():

        raise FileNotFoundError(
            folder
        )


    for p in sorted(

        folder.glob(
            "results_sigma_0.0_trial_*_dends_*_soma_*.pkl"
        )

    ):

        m = RESULT_PATTERN.fullmatch(
            p.name
        )


        if not m:

            continue


        trial, D, S = map(
            int,
            m.groups(),
        )


        with open(
            p,
            "rb",
        ) as f:

            out = pickle.load(
                f
            )


        rows.append({

            "model":
                model_name,

            "trial":
                trial,

            "num_dends":
                D,

            "num_soma":
                S,

            "fraction":
                fraction,

            "lambda":
                lam,

            "test_loss":
                extract_scalar(
                    out[
                        "test_loss"
                    ]
                ),

            "test_acc":
                (
                    extract_scalar(
                        out[
                            "test_acc"
                        ]
                    )
                    * 100
                ),

            "trainable_params":
                nparams(
                    D,
                    S,
                ),
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# LOAD PERFORMANCE FOR EVERY MODEL
# ============================================================

performance_dfs = []


for model_name, info in MODEL_INFO.items():

    df_model = (
        load_results_from_folder(

            model_name=
                model_name,

            folder=
                info[
                    "folder"
                ],

            fraction=
                info[
                    "fraction"
                ],

            lam=
                info[
                    "lambda"
                ],
        )
    )


    performance_dfs.append(
        df_model
    )


DF_PERF = pd.concat(
    performance_dfs,
    ignore_index=True,
)


# ============================================================
# PERFORMANCE PLOT COLORS
# ============================================================

def get_all_model_color(
    model_name,
):

    if model_name == "dANN-LRF":

        return COLOR_LRF


    if model_name == "dANN-C 10%":

        return COUPLING_COLORS[
            10
        ]


    if model_name == "dANN-C 20%":

        return COUPLING_COLORS[
            20
        ]


    if model_name == "dANN-C 30%":

        return COUPLING_COLORS[
            30
        ]


    if model_name == "dANN-C 40%":

        return COUPLING_COLORS[
            40
        ]


    if model_name == "dANN-C λ=0.05":

        return LAMBDA_COLORS[
            0.05
        ]


    if model_name == "dANN-C λ=0.20":

        return LAMBDA_COLORS[
            0.20
        ]


    return "0.5"



def get_coupling_color(
    model_name,
):

    if model_name == "dANN-LRF":

        return COLOR_LRF


    return COUPLING_COLORS[
        MODEL_INFO[
            model_name
        ][
            "fraction"
        ]
    ]



def get_lambda_color(
    model_name,
):

    if model_name == "dANN-LRF":

        return COLOR_LRF


    return LAMBDA_COLORS[
        MODEL_INFO[
            model_name
        ][
            "lambda"
        ]
    ]


# ============================================================
# FIGURE 2 PERFORMANCE
# ============================================================

def plot_performance_figure(
    df,
    models,
    filename,
    color_function,
):

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            9,
            7.5,
        ),
    )


    # ========================================================
    # TEST LOSS vs PARAMETERS
    # ========================================================

    ax = axes[
        0,
        0
    ]


    for model_name in models:

        d = (

            df[
                df[
                    "model"
                ] == model_name
            ]

            .groupby(
                "trainable_params",
                as_index=False,
            )

            .agg(
                test_loss=(
                    "test_loss",
                    "mean",
                )
            )

            .sort_values(
                "trainable_params"
            )
        )


        ax.plot(

            d[
                "trainable_params"
            ],

            d[
                "test_loss"
            ],

            marker=
                "o",

            markersize=
                4,

            linewidth=
                1.6,

            color=
                color_function(
                    model_name
                ),

            label=
                model_name,
        )


    if VANN_REF is not None:

        ax.axhline(

            VANN_REF[
                "best_loss"
            ],

            ls=
                "--",

            color=
                "0.6",

            lw=
                1,
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


    # ========================================================
    # TEST ACCURACY vs PARAMETERS
    # ========================================================

    ax = axes[
        0,
        1
    ]


    for model_name in models:

        d = (

            df[
                df[
                    "model"
                ] == model_name
            ]

            .groupby(
                "trainable_params",
                as_index=False,
            )

            .agg(
                test_acc=(
                    "test_acc",
                    "mean",
                )
            )

            .sort_values(
                "trainable_params"
            )
        )


        ax.plot(

            d[
                "trainable_params"
            ],

            d[
                "test_acc"
            ],

            marker=
                "o",

            markersize=
                4,

            linewidth=
                1.6,

            color=
                color_function(
                    model_name
                ),

            label=
                model_name,
        )


    if VANN_REF is not None:

        ax.axhline(

            VANN_REF[
                "best_acc"
            ],

            ls=
                "--",

            color=
                "0.6",

            lw=
                1,
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


    # ========================================================
    # D-SWEEP — S > 200
    # ========================================================

    dsub = df[
        df[
            "num_soma"
        ] > 200
    ].copy()


    # ========================================================
    # LOSS vs D
    # ========================================================

    ax = axes[
        1,
        0
    ]


    for model_name in models:

        for S in sorted(

            dsub[
                "num_soma"
            ].unique()

        ):

            d = (

                dsub[
                    (
                        dsub[
                            "model"
                        ] == model_name
                    )

                    &

                    (
                        dsub[
                            "num_soma"
                        ] == S
                    )
                ]

                .groupby(
                    "num_dends",
                    as_index=False,
                )

                .agg(
                    test_loss=(
                        "test_loss",
                        "mean",
                    )
                )

                .sort_values(
                    "num_dends"
                )
            )


            if len(d) == 0:

                continue


            marker = (

                "o"
                if S == 256

                else "s"
            )


            ax.plot(

                d[
                    "num_dends"
                ],

                d[
                    "test_loss"
                ],

                marker=
                    marker,

                markersize=
                    4,

                linewidth=
                    1.5,

                color=
                    color_function(
                        model_name
                    ),
            )


    if VANN_REF is not None:

        ax.axhline(

            VANN_REF[
                "best_loss"
            ],

            ls=
                "--",

            color=
                "0.6",

            lw=
                1,
        )


    ax.set_xscale(
        "log",
        base=2,
    )


    ax.set_xticks(
        [
            2,
            4,
            8,
            16,
            32,
            64,
        ]
    )


    ax.set_xticklabels(
        [
            2,
            4,
            8,
            16,
            32,
            64,
        ]
    )


    ax.set_xlabel(
        "Dendrites per soma"
    )


    ax.set_ylabel(
        "Test loss"
    )


    # ========================================================
    # ACCURACY vs D
    # ========================================================

    ax = axes[
        1,
        1
    ]


    for model_name in models:

        for S in sorted(

            dsub[
                "num_soma"
            ].unique()

        ):

            d = (

                dsub[
                    (
                        dsub[
                            "model"
                        ] == model_name
                    )

                    &

                    (
                        dsub[
                            "num_soma"
                        ] == S
                    )
                ]

                .groupby(
                    "num_dends",
                    as_index=False,
                )

                .agg(
                    test_acc=(
                        "test_acc",
                        "mean",
                    )
                )

                .sort_values(
                    "num_dends"
                )
            )


            if len(d) == 0:

                continue


            marker = (

                "o"
                if S == 256

                else "s"
            )


            ax.plot(

                d[
                    "num_dends"
                ],

                d[
                    "test_acc"
                ],

                marker=
                    marker,

                markersize=
                    4,

                linewidth=
                    1.5,

                color=
                    color_function(
                        model_name
                    ),
            )


    if VANN_REF is not None:

        ax.axhline(

            VANN_REF[
                "best_acc"
            ],

            ls=
                "--",

            color=
                "0.6",

            lw=
                1,
        )


    ax.set_xscale(
        "log",
        base=2,
    )


    ax.set_xticks(
        [
            2,
            4,
            8,
            16,
            32,
            64,
        ]
    )


    ax.set_xticklabels(
        [
            2,
            4,
            8,
            16,
            32,
            64,
        ]
    )


    ax.set_xlabel(
        "Dendrites per soma"
    )


    ax.set_ylabel(
        "Test accuracy (%)"
    )


    # ========================================================
    # LEGEND
    # ========================================================

    handles = [

        Line2D(

            [0],
            [0],

            color=
                color_function(
                    model_name
                ),

            marker=
                "o",

            lw=
                1.8,

            label=
                model_name,
        )

        for model_name in models
    ]


    axes[
        0,
        1
    ].legend(

        handles=
            handles,

        frameon=
            False,

        fontsize=
            8,
    )


    sns.despine()

    fig.tight_layout()


    fig.savefig(

        OUT
        / filename,

        bbox_inches=
            "tight",
    )


    plt.close(
        fig
    )


# ============================================================
# PERFORMANCE FIGURES
# ============================================================

plot_performance_figure(

    df=
        DF_PERF,

    models=
        ALL_MODELS,

    filename=
        "figure2_all_models.pdf",

    color_function=
        get_all_model_color,
)


plot_performance_figure(

    df=
        DF_PERF[
            DF_PERF[
                "model"
            ].isin(
                COUPLING_MODELS
            )
        ],

    models=
        COUPLING_MODELS,

    filename=
        "figure2_coupling_sweep.pdf",

    color_function=
        get_coupling_color,
)


plot_performance_figure(

    df=
        DF_PERF[
            DF_PERF[
                "model"
            ].isin(
                LAMBDA_MODELS
            )
        ],

    models=
        LAMBDA_MODELS,

    filename=
        "figure2_lambda_sweep.pdf",

    color_function=
        get_lambda_color,
)


# ============================================================
# FIND AVAILABLE WEIGHT TRIALS
# ============================================================

def find_available_trials(
    folder,
    D,
    S,
):

    trials = []


    for p in folder.glob(

        f"model_sigma_{SIGMA}"
        f"_trial_*"
        f"_dends_{D}"
        f"_soma_{S}.keras"

    ):

        m = MODEL_PATTERN.fullmatch(
            p.name
        )


        if not m:

            continue


        trial, d_, s_ = map(
            int,
            m.groups(),
        )


        if (
            d_ == D
            and
            s_ == S
        ):

            trials.append(
                trial
            )


    return sorted(
        set(
            trials
        )
    )


# ============================================================
# PRINT TRIAL AVAILABILITY
# ============================================================

print(
    "\n=============================================="
)

print(
    "WEIGHT ANALYSIS TRIAL AVAILABILITY"
)

print(
    "=============================================="
)


for model_name in ALL_MODELS:

    trials = (
        find_available_trials(

            MODEL_INFO[
                model_name
            ][
                "folder"
            ],

            BEST_D,
            BEST_S,
        )
    )


    print(

        f"{model_name:20s}: "
        f"n={len(trials):2d} "
        f"{trials}"
    )


# ============================================================
# LOAD ONE MODEL'S WEIGHTS
# ============================================================

def load_grouped_weights(
    model_name,
    trial,
    D,
    S,
):

    info = MODEL_INFO[
        model_name
    ]


    fname = (

        info[
            "folder"
        ]

        /

        (
            f"model_sigma_{SIGMA}"
            f"_trial_{trial}"
            f"_dends_{D}"
            f"_soma_{S}.keras"
        )
    )


    if not fname.exists():

        raise FileNotFoundError(
            fname
        )


    model = keras.models.load_model(

        fname,

        custom_objects={
            "DendriticCoupling":
                DendriticCoupling
        },
    )


    W_input = (

        model
        .get_layer(
            "dend_1"
        )
        .get_weights()[0]
    )


    W_soma = (

        model
        .get_layer(
            "soma_1"
        )
        .get_weights()[0]
    )


    W_output = (

        model
        .get_layer(
            "output"
        )
        .get_weights()[0]
    )


    # ========================================================
    # dANN-LRF
    # ========================================================

    if model_name == "dANN-LRF":

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


        result = {

            "dANN-LRF": {

                "Input → dendrite":
                    remove_zeros(
                        W_input
                    ),

                "Dendrite → soma":
                    remove_zeros(
                        np.concatenate(
                            soma_weights
                        )
                    ),

                "Soma → output":
                    remove_zeros(
                        W_output
                    ),
            }
        }


        keras.backend.clear_session()


        return result


    # ========================================================
    # dANN-C
    # ========================================================

    fraction = (

        MODEL_INFO[
            model_name
        ][
            "fraction"
        ]

        / 100.0
    )


    n_coupled = int(

        round(
            S
            * fraction
        )
    )


    grouped = {

        "Uncoupled": {

            layer: []

            for layer in LAYERS
        },

        "Coupled": {

            layer: []

            for layer in LAYERS
        },
    }


    for soma_idx in range(
        S
    ):

        group = (

            "Coupled"

            if soma_idx < n_coupled

            else "Uncoupled"
        )


        d0 = (
            soma_idx
            * D
        )


        d1 = (
            (soma_idx + 1)
            * D
        )


        # ----------------------------------------------------
        # Input -> dendrite
        # ----------------------------------------------------

        grouped[
            group
        ][
            "Input → dendrite"
        ].append(

            remove_zeros(

                W_input[
                    :,
                    d0:d1
                ]
            )
        )


        # ----------------------------------------------------
        # Dendrite -> soma
        # ----------------------------------------------------

        grouped[
            group
        ][
            "Dendrite → soma"
        ].append(

            remove_zeros(

                W_soma[
                    d0:d1,
                    soma_idx
                ]
            )
        )


        # ----------------------------------------------------
        # Soma -> output
        # ----------------------------------------------------

        grouped[
            group
        ][
            "Soma → output"
        ].append(

            remove_zeros(

                W_output[
                    soma_idx,
                    :
                ]
            )
        )


    for group in [
        "Uncoupled",
        "Coupled",
    ]:

        for layer in LAYERS:

            arrays = [

                x

                for x in grouped[
                    group
                ][
                    layer
                ]

                if len(x)
            ]


            grouped[
                group
            ][
                layer
            ] = (

                np.concatenate(
                    arrays
                )

                if arrays

                else np.array([])
            )


    keras.backend.clear_session()


    return grouped


# ============================================================
# LOAD WEIGHTS FROM ALL AVAILABLE TRIALS
# ============================================================

RAW_WEIGHTS = {}

SUMMARY_ROWS = []


for model_name in ALL_MODELS:

    trials = (
        find_available_trials(

            MODEL_INFO[
                model_name
            ][
                "folder"
            ],

            BEST_D,
            BEST_S,
        )
    )


    if len(trials) == 0:

        print(
            f"WARNING: no trials found for {model_name}"
        )

        continue


    # ========================================================
    # dANN-LRF
    # ========================================================

    if model_name == "dANN-LRF":

        RAW_WEIGHTS[
            model_name
        ] = {

            layer: []

            for layer in LAYERS
        }


        for trial in trials:

            out = (
                load_grouped_weights(

                    model_name,
                    trial,
                    BEST_D,
                    BEST_S,
                )[
                    "dANN-LRF"
                ]
            )


            for layer in LAYERS:

                weights = out[
                    layer
                ]


                RAW_WEIGHTS[
                    model_name
                ][
                    layer
                ].append(
                    weights
                )


                stats = (
                    summarize_weights(
                        weights
                    )
                )


                SUMMARY_ROWS.append({

                    "model":
                        model_name,

                    "trial":
                        trial,

                    "fraction":
                        0,

                    "lambda":
                        0.0,

                    "group":
                        "dANN-LRF",

                    "layer":
                        layer,

                    **stats,
                })


        for layer in LAYERS:

            RAW_WEIGHTS[
                model_name
            ][
                layer
            ] = np.concatenate(

                RAW_WEIGHTS[
                    model_name
                ][
                    layer
                ]
            )


    # ========================================================
    # dANN-C
    # ========================================================

    else:

        RAW_WEIGHTS[
            model_name
        ] = {

            "Uncoupled": {

                layer: []

                for layer in LAYERS
            },

            "Coupled": {

                layer: []

                for layer in LAYERS
            },
        }


        for trial in trials:

            out = (
                load_grouped_weights(

                    model_name,
                    trial,
                    BEST_D,
                    BEST_S,
                )
            )


            for group in [
                "Uncoupled",
                "Coupled",
            ]:

                for layer in LAYERS:

                    weights = (

                        out[
                            group
                        ][
                            layer
                        ]
                    )


                    RAW_WEIGHTS[
                        model_name
                    ][
                        group
                    ][
                        layer
                    ].append(
                        weights
                    )


                    stats = (
                        summarize_weights(
                            weights
                        )
                    )


                    SUMMARY_ROWS.append({

                        "model":
                            model_name,

                        "trial":
                            trial,

                        "fraction":
                            MODEL_INFO[
                                model_name
                            ][
                                "fraction"
                            ],

                        "lambda":
                            MODEL_INFO[
                                model_name
                            ][
                                "lambda"
                            ],

                        "group":
                            group,

                        "layer":
                            layer,

                        **stats,
                    })


        for group in [
            "Uncoupled",
            "Coupled",
        ]:

            for layer in LAYERS:

                RAW_WEIGHTS[
                    model_name
                ][
                    group
                ][
                    layer
                ] = np.concatenate(

                    RAW_WEIGHTS[
                        model_name
                    ][
                        group
                    ][
                        layer
                    ]
                )


DF_SUMMARY = pd.DataFrame(
    SUMMARY_ROWS
)


# ============================================================
# SAVE WEIGHT SUMMARY DATA
# ============================================================

DF_SUMMARY.to_csv(

    OUT
    / "weight_metrics_best_models.csv",

    index=False,
)


# ============================================================
# FIGURE 5
#
# 3 ROWS x 5 COLUMNS
#
# COL 1:
# dANN-LRF
#
# COL 2:
# coupling fraction — UNCOUPLED
#
# COL 3:
# coupling fraction — COUPLED
#
# COL 4:
# lambda — UNCOUPLED
#
# COL 5:
# lambda — COUPLED
# ============================================================

fig, axes = plt.subplots(

    3,
    5,

    figsize=(
        17,
        9,
    ),

    sharey=
        "row",
)


# ============================================================
# MODEL LISTS FOR DISTRIBUTIONS
# ============================================================

COUPLING_DISTRIBUTION_MODELS = [

    "dANN-C 10%",
    "dANN-C 20%",
    "dANN-C 30%",
    "dANN-C 40%",
]


LAMBDA_DISTRIBUTION_MODELS = [

    "dANN-C λ=0.05",
    "dANN-C 20%",
    "dANN-C λ=0.20",
]


# ============================================================
# DRAW EACH LAYER
# ============================================================

for row, layer in enumerate(
    LAYERS
):

    # ========================================================
    # COLUMN 1 — dANN-LRF
    # ========================================================

    ax = axes[
        row,
        0
    ]


    sns.kdeplot(

        x=
            RAW_WEIGHTS[
                "dANN-LRF"
            ][
                layer
            ],

        ax=
            ax,

        color=
            COLOR_LRF,

        linewidth=
            2,

        fill=
            False,

        common_norm=
            False,
    )


    ax.axvline(

        0,

        color=
            "0.75",

        ls=
            "--",

        lw=
            0.8,
    )


    if row == 0:

        ax.set_title(
            "dANN-LRF"
        )


    ax.set_ylabel(

        f"{layer}\n"
        "Probability density"
    )


    # ========================================================
    # COLUMN 2 — FRACTION UNCOUPLED
    # ========================================================

    ax = axes[
        row,
        1
    ]


    for model_name in (
        COUPLING_DISTRIBUTION_MODELS
    ):

        fraction = (
            MODEL_INFO[
                model_name
            ][
                "fraction"
            ]
        )


        sns.kdeplot(

            x=
                RAW_WEIGHTS[
                    model_name
                ][
                    "Uncoupled"
                ][
                    layer
                ],

            ax=
                ax,

            color=
                COUPLING_COLORS[
                    fraction
                ],

            linewidth=
                1.8,

            fill=
                False,

            common_norm=
                False,

            label=
                f"{fraction}%",
        )


    ax.axvline(

        0,

        color=
            "0.75",

        ls=
            "--",

        lw=
            0.8,
    )


    if row == 0:

        ax.set_title(
            "Coupling proportion\nUncoupled"
        )


    ax.set_ylabel("")


    # ========================================================
    # COLUMN 3 — FRACTION COUPLED
    # ========================================================

    ax = axes[
        row,
        2
    ]


    for model_name in (
        COUPLING_DISTRIBUTION_MODELS
    ):

        fraction = (
            MODEL_INFO[
                model_name
            ][
                "fraction"
            ]
        )


        sns.kdeplot(

            x=
                RAW_WEIGHTS[
                    model_name
                ][
                    "Coupled"
                ][
                    layer
                ],

            ax=
                ax,

            color=
                COUPLING_COLORS[
                    fraction
                ],

            linewidth=
                1.8,

            fill=
                False,

            common_norm=
                False,

            label=
                f"{fraction}%",
        )


    ax.axvline(

        0,

        color=
            "0.75",

        ls=
            "--",

        lw=
            0.8,
    )


    if row == 0:

        ax.set_title(
            "Coupling proportion\nCoupled"
        )


    ax.set_ylabel("")


    # ========================================================
    # COLUMN 4 — LAMBDA UNCOUPLED
    # ========================================================

    ax = axes[
        row,
        3
    ]


    for model_name in (
        LAMBDA_DISTRIBUTION_MODELS
    ):

        lam = (
            MODEL_INFO[
                model_name
            ][
                "lambda"
            ]
        )


        sns.kdeplot(

            x=
                RAW_WEIGHTS[
                    model_name
                ][
                    "Uncoupled"
                ][
                    layer
                ],

            ax=
                ax,

            color=
                LAMBDA_COLORS[
                    lam
                ],

            linewidth=
                1.8,

            fill=
                False,

            common_norm=
                False,

            label=
                f"λ={lam:.2f}",
        )


    ax.axvline(

        0,

        color=
            "0.75",

        ls=
            "--",

        lw=
            0.8,
    )


    if row == 0:

        ax.set_title(
            "Coupling λ\nUncoupled"
        )


    ax.set_ylabel("")


    # ========================================================
    # COLUMN 5 — LAMBDA COUPLED
    # ========================================================

    ax = axes[
        row,
        4
    ]


    for model_name in (
        LAMBDA_DISTRIBUTION_MODELS
    ):

        lam = (
            MODEL_INFO[
                model_name
            ][
                "lambda"
            ]
        )


        sns.kdeplot(

            x=
                RAW_WEIGHTS[
                    model_name
                ][
                    "Coupled"
                ][
                    layer
                ],

            ax=
                ax,

            color=
                LAMBDA_COLORS[
                    lam
                ],

            linewidth=
                1.8,

            fill=
                False,

            common_norm=
                False,

            label=
                f"λ={lam:.2f}",
        )


    ax.axvline(

        0,

        color=
            "0.75",

        ls=
            "--",

        lw=
            0.8,
    )


    if row == 0:

        ax.set_title(
            "Coupling λ\nCoupled"
        )


    ax.set_ylabel("")


    # ========================================================
    # X LABELS
    # ========================================================

    for col in range(
        5
    ):

        if row == 2:

            axes[
                row,
                col
            ].set_xlabel(
                "Weight"
            )

        else:

            axes[
                row,
                col
            ].set_xlabel("")


# ============================================================
# DISTRIBUTION LEGENDS
# ============================================================

axes[
    0,
    1
].legend(

    title=
        "Coupled somata",

    frameon=
        False,

    fontsize=
        7,
)


axes[
    0,
    2
].legend(

    title=
        "Coupled somata",

    frameon=
        False,

    fontsize=
        7,
)


axes[
    0,
    3
].legend(

    title=
        "λ",

    frameon=
        False,

    fontsize=
        7,
)


axes[
    0,
    4
].legend(

    title=
        "λ",

    frameon=
        False,

    fontsize=
        7,
)


sns.despine()

fig.tight_layout()


fig.savefig(

    OUT
    / "figure5_weight_distributions_summary.pdf",

    bbox_inches=
        "tight",
)


plt.close(
    fig
)


# ============================================================
# WEIGHT-METRIC PLOT HELPERS
#
# Coupling fraction:
#
# x = 0, 10, 20, 30, 40
#
# lambda:
#
# x = 0, 0.05, 0.10, 0.20
#
# At x=0:
# black dANN-LRF point ± SEM
#
# Coupled and uncoupled trajectories begin at actual
# non-zero dANN-C conditions.
# ============================================================

def plot_coupling_metric(
    ax,
    layer,
    metric,
    ylabel="",
):

    d = DF_SUMMARY[
        DF_SUMMARY[
            "layer"
        ] == layer
    ].copy()


    # ========================================================
    # dANN-LRF at x = 0
    # ========================================================

    lrf_values = (

        d[
            (
                d[
                    "model"
                ] == "dANN-LRF"
            )

            &

            (
                d[
                    "group"
                ] == "dANN-LRF"
            )
        ][
            metric
        ]

        .dropna()

        .to_numpy()
    )


    lrf_mean, lrf_sem = (
        mean_sem(
            lrf_values
        )
    )


    ax.errorbar(

        0,

        lrf_mean,

        yerr=
            lrf_sem,

        fmt=
            "o",

        markersize=
            6,

        capsize=
            3,

        linewidth=
            1.5,

        color=
            GROUP_COLORS[
                "dANN-LRF"
            ],

        label=
            "dANN-LRF",

        zorder=
            10,
    )


    all_values = list(
        lrf_values
    )


    # ========================================================
    # dANN-C trajectories
    # ========================================================

    fractions = [
        10,
        20,
        30,
        40,
    ]


    for group in [
        "Uncoupled",
        "Coupled",
    ]:

        means = []
        sems = []


        for fraction in fractions:

            vals = (

                d[
                    (
                        d[
                            "fraction"
                        ] == fraction
                    )

                    &

                    (
                        d[
                            "lambda"
                        ] == 0.10
                    )

                    &

                    (
                        d[
                            "group"
                        ] == group
                    )
                ][
                    metric
                ]

                .dropna()

                .to_numpy()
            )


            mean, sem = (
                mean_sem(
                    vals
                )
            )


            means.append(
                mean
            )

            sems.append(
                sem
            )


            all_values.extend(
                vals.tolist()
            )


        means = np.asarray(
            means,
            dtype=float,
        )


        sems = np.asarray(
            sems,
            dtype=float,
        )


        # ----------------------------------------------------
        # SEM
        # ----------------------------------------------------

        ax.fill_between(

            fractions,

            means - sems,

            means + sems,

            color=
                GROUP_COLORS[
                    group
                ],

            alpha=
                0.15,

            linewidth=
                0,
        )


        # ----------------------------------------------------
        # Mean line
        # ----------------------------------------------------

        ax.plot(

            fractions,

            means,

            marker=
                "o",

            markersize=
                5,

            linewidth=
                1.8,

            color=
                GROUP_COLORS[
                    group
                ],

            label=
                group,
        )


    ax.set_xticks(
        [
            0,
            10,
            20,
            30,
            40,
        ]
    )


    ax.set_xticklabels(
        [
            "0",
            "10",
            "20",
            "30",
            "40",
        ]
    )


    ax.set_xlabel(
        "Coupled somata (%)"
    )


    ax.set_ylabel(
        ylabel
    )


    zoom_axis(
        ax,
        all_values,
    )


# ============================================================
# LAMBDA METRIC
#
# IMPORTANT:
# dANN-LRF is now a point at λ = 0.
# ============================================================

def plot_lambda_metric(
    ax,
    layer,
    metric,
    ylabel="",
):

    d = DF_SUMMARY[
        DF_SUMMARY[
            "layer"
        ] == layer
    ].copy()


    # ========================================================
    # dANN-LRF at λ = 0
    # ========================================================

    lrf_values = (

        d[
            (
                d[
                    "model"
                ] == "dANN-LRF"
            )

            &

            (
                d[
                    "group"
                ] == "dANN-LRF"
            )
        ][
            metric
        ]

        .dropna()

        .to_numpy()
    )


    lrf_mean, lrf_sem = (
        mean_sem(
            lrf_values
        )
    )


    ax.errorbar(

        0,

        lrf_mean,

        yerr=
            lrf_sem,

        fmt=
            "o",

        markersize=
            6,

        capsize=
            3,

        linewidth=
            1.5,

        color=
            GROUP_COLORS[
                "dANN-LRF"
            ],

        label=
            "dANN-LRF",

        zorder=
            10,
    )


    all_values = list(
        lrf_values
    )


    # ========================================================
    # dANN-C lambda trajectories
    # ========================================================

    lambdas = [
        0.05,
        0.10,
        0.20,
    ]


    for group in [
        "Uncoupled",
        "Coupled",
    ]:

        means = []
        sems = []


        for lam in lambdas:

            vals = (

                d[
                    (
                        d[
                            "fraction"
                        ] == 20
                    )

                    &

                    (
                        d[
                            "lambda"
                        ] == lam
                    )

                    &

                    (
                        d[
                            "group"
                        ] == group
                    )
                ][
                    metric
                ]

                .dropna()

                .to_numpy()
            )


            mean, sem = (
                mean_sem(
                    vals
                )
            )


            means.append(
                mean
            )

            sems.append(
                sem
            )


            all_values.extend(
                vals.tolist()
            )


        means = np.asarray(
            means,
            dtype=float,
        )


        sems = np.asarray(
            sems,
            dtype=float,
        )


        # ----------------------------------------------------
        # SEM
        # ----------------------------------------------------

        ax.fill_between(

            lambdas,

            means - sems,

            means + sems,

            color=
                GROUP_COLORS[
                    group
                ],

            alpha=
                0.15,

            linewidth=
                0,
        )


        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        ax.plot(

            lambdas,

            means,

            marker=
                "o",

            markersize=
                5,

            linewidth=
                1.8,

            color=
                GROUP_COLORS[
                    group
                ],

            label=
                group,
        )


    ax.set_xticks(
        [
            0,
            0.05,
            0.10,
            0.20,
        ]
    )


    ax.set_xticklabels(
        [
            "0",
            "0.05",
            "0.10",
            "0.20",
        ]
    )


    ax.set_xlabel(
        "Coupling λ"
    )


    ax.set_ylabel(
        ylabel
    )


    zoom_axis(
        ax,
        all_values,
    )


# ============================================================
# WEIGHT METRIC FIGURE
#
# 3 ROWS:
#
# Input → dendrite
# Dendrite → soma
# Soma → output
#
#
# 6 COLUMNS:
#
# COUPLING PROPORTION
# 1 positive fraction
# 2 positive strength
# 3 negative strength
#
# LAMBDA
# 4 positive fraction
# 5 positive strength
# 6 negative strength
# ============================================================

fig, axes = plt.subplots(

    3,
    6,

    figsize=(
        20,
        9,
    ),
)


METRICS = [

    (
        "fraction_positive",
        "Positive fraction",
        "Fraction positive",
    ),

    (
        "mean_positive",
        "Positive strength",
        "Mean positive weight",
    ),

    (
        "mean_negative",
        "Negative strength",
        "Mean negative weight",
    ),
]


# ============================================================
# DRAW
# ============================================================

for row, layer in enumerate(
    LAYERS
):

    # ========================================================
    # COUPLING PROPORTION — columns 0:3
    # ========================================================

    for metric_index, (
        metric,
        title,
        ylabel,
    ) in enumerate(
        METRICS
    ):

        col = metric_index


        plot_coupling_metric(

            ax=
                axes[
                    row,
                    col
                ],

            layer=
                layer,

            metric=
                metric,

            ylabel=
                (
                    f"{layer}\n{ylabel}"

                    if col == 0

                    else ""
                ),
        )


        if row == 0:

            axes[
                row,
                col
            ].set_title(

                "Coupling proportion\n"
                f"{title}"
            )


        if row != 2:

            axes[
                row,
                col
            ].set_xlabel("")


    # ========================================================
    # LAMBDA — columns 3:6
    # ========================================================

    for metric_index, (
        metric,
        title,
        ylabel,
    ) in enumerate(
        METRICS
    ):

        col = (
            3
            + metric_index
        )


        plot_lambda_metric(

            ax=
                axes[
                    row,
                    col
                ],

            layer=
                layer,

            metric=
                metric,

            ylabel="",
        )


        if row == 0:

            axes[
                row,
                col
            ].set_title(

                "Coupling λ\n"
                f"{title}"
            )


        if row != 2:

            axes[
                row,
                col
            ].set_xlabel("")


# ============================================================
# METRIC LEGEND
# ============================================================

metric_handles = [

    Line2D(

        [0],
        [0],

        color=
            GROUP_COLORS[
                "dANN-LRF"
            ],

        marker=
            "o",

        linestyle=
            "None",

        markersize=
            6,

        label=
            "dANN-LRF",
    ),


    Line2D(

        [0],
        [0],

        color=
            GROUP_COLORS[
                "Uncoupled"
            ],

        marker=
            "o",

        linewidth=
            1.8,

        label=
            "Uncoupled",
    ),


    Line2D(

        [0],
        [0],

        color=
            GROUP_COLORS[
                "Coupled"
            ],

        marker=
            "o",

        linewidth=
            1.8,

        label=
            "Coupled",
    ),
]


axes[
    0,
    2
].legend(

    handles=
        metric_handles,

    frameon=
        False,

    fontsize=
        8,
)


axes[
    0,
    5
].legend(

    handles=
        metric_handles,

    frameon=
        False,

    fontsize=
        8,
)


sns.despine()

fig.tight_layout()


fig.savefig(

    OUT
    / "weight_metrics_evolution_best_models.pdf",

    bbox_inches=
        "tight",
)


plt.close(
    fig
)

# weight_metrics_evolution_dendrites.py

import re
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import keras

from opt import DendriticCoupling
from plotting_functions import my_style


# ============================================================
# PATHS
# ============================================================

ROOT = pathlib.Path.home() / "dendritic_anns"

DATA = (
    ROOT
    / "DATA"
    / "results_fmnist_1_layer"
)

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

SIGMA = 0.0

SOMA = 512

DENDS = [
    2,
    4,
    8,
    16,
    32,
    64,
]


# ============================================================
# MODEL PATHS
# ============================================================

LRF_FOLDER = (
    DATA
    / "dend_ann_local_rfs"
)

C_FOLDER = (
    DATA_C
    / "dend_ann_local_rfs_C"
)


# ============================================================
# dANN-C PARAMETERS
# ============================================================

COUPLING_FRACTION = 0.20

COUPLING_LAMBDA = 0.10


# ============================================================
# LAYERS
# ============================================================

LAYERS = [
    "Input → dendrite",
    "Dendrite → soma",
    "Soma → output",
]


# ============================================================
# COLORS
# ============================================================

COLORS = {

    "dANN-LRF":
        "#000000",

    "Uncoupled":
        "#969696",

    "Coupled":
        "#8c2d8c",
}


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
# FILE PATTERN
# ============================================================

MODEL_PATTERN = re.compile(

    r"model_sigma_0\.0"
    r"_trial_(\d+)"
    r"_dends_(\d+)"
    r"_soma_(\d+)\.keras"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def remove_zeros(
    weights,
):

    weights = np.asarray(
        weights,
        dtype=float,
    ).ravel()


    return weights[
        weights != 0
    ]



def summarize_weights(
    weights,
):

    weights = remove_zeros(
        weights
    )


    if len(weights) == 0:

        return {

            "fraction_positive":
                np.nan,

            "mean_positive":
                np.nan,

            "mean_negative":
                np.nan,
        }


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

        "mean_positive":
            (
                np.mean(
                    positive
                )

                if len(positive)

                else np.nan
            ),

        "mean_negative":
            (
                np.mean(
                    negative
                )

                if len(negative)

                else np.nan
            ),
    }



def mean_sem(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(values) == 0:

        return (
            np.nan,
            np.nan,
        )


    mean = np.mean(
        values
    )


    if len(values) > 1:

        sem = (

            np.std(
                values,
                ddof=1,
            )

            /

            np.sqrt(
                len(values)
            )
        )

    else:

        sem = np.nan


    return (
        mean,
        sem,
    )



def zoom_axis(
    ax,
    values,
    margin_fraction=0.20,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(values) == 0:

        return


    ymin = np.min(
        values
    )


    ymax = np.max(
        values
    )


    span = (
        ymax
        - ymin
    )


    if span == 0:

        span = (

            abs(ymin)
            * 0.10

            if ymin != 0

            else 0.1
        )


    margin = (
        span
        * margin_fraction
    )


    ax.set_ylim(

        ymin
        - margin,

        ymax
        + margin,
    )


# ============================================================
# FIND ALL AVAILABLE TRIALS FOR ONE ARCHITECTURE
# ============================================================

def find_available_trials(
    folder,
    D,
    S,
):

    trials = []


    pattern = (

        f"model_sigma_{SIGMA}"
        f"_trial_*"
        f"_dends_{D}"
        f"_soma_{S}.keras"
    )


    for p in folder.glob(
        pattern
    ):

        m = MODEL_PATTERN.fullmatch(
            p.name
        )


        if not m:

            continue


        trial, d_, s_ = map(
            int,
            m.groups(),
        )


        if (
            d_ == D
            and
            s_ == S
        ):

            trials.append(
                trial
            )


    return sorted(
        set(
            trials
        )
    )


# ============================================================
# MODEL PATH
# ============================================================

def model_file(
    folder,
    trial,
    D,
    S,
):

    return (

        folder
        /

        (
            f"model_sigma_{SIGMA}"
            f"_trial_{trial}"
            f"_dends_{D}"
            f"_soma_{S}.keras"
        )
    )


# ============================================================
# LOAD dANN-LRF WEIGHTS
# ============================================================

def load_lrf_weights(
    trial,
    D,
    S,
):

    fname = model_file(
        LRF_FOLDER,
        trial,
        D,
        S,
    )


    if not fname.exists():

        raise FileNotFoundError(
            fname
        )


    model = keras.models.load_model(
        fname
    )


    # --------------------------------------------------------
    # Input -> dendrite
    # --------------------------------------------------------

    W_input = (

        model
        .get_layer(
            "dend_1"
        )
        .get_weights()[0]
    )


    # --------------------------------------------------------
    # Dendrite -> soma
    # --------------------------------------------------------

    W_soma = (

        model
        .get_layer(
            "soma_1"
        )
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


    # --------------------------------------------------------
    # Soma -> output
    # --------------------------------------------------------

    W_output = (

        model
        .get_layer(
            "output"
        )
        .get_weights()[0]
    )


    result = {

        "Input → dendrite":
            remove_zeros(
                W_input
            ),

        "Dendrite → soma":
            remove_zeros(
                np.concatenate(
                    soma_weights
                )
            ),

        "Soma → output":
            remove_zeros(
                W_output
            ),
    }


    keras.backend.clear_session()


    return result


# ============================================================
# LOAD dANN-C WEIGHTS
#
# Split into:
# - Coupled
# - Uncoupled
# ============================================================

def load_dannc_weights(
    trial,
    D,
    S,
):

    fname = model_file(
        C_FOLDER,
        trial,
        D,
        S,
    )


    if not fname.exists():

        raise FileNotFoundError(
            fname
        )


    model = keras.models.load_model(

        fname,

        custom_objects={
            "DendriticCoupling":
                DendriticCoupling
        },
    )


    # --------------------------------------------------------
    # Weight matrices
    # --------------------------------------------------------

    W_input = (

        model
        .get_layer(
            "dend_1"
        )
        .get_weights()[0]
    )


    W_soma = (

        model
        .get_layer(
            "soma_1"
        )
        .get_weights()[0]
    )


    W_output = (

        model
        .get_layer(
            "output"
        )
        .get_weights()[0]
    )


    # --------------------------------------------------------
    # Number of coupled somata
    # --------------------------------------------------------

    n_coupled = int(

        round(
            S
            * COUPLING_FRACTION
        )
    )


    grouped = {

        "Uncoupled": {

            layer: []

            for layer in LAYERS
        },

        "Coupled": {

            layer: []

            for layer in LAYERS
        },
    }


    # --------------------------------------------------------
    # Assign weights by parent soma
    # --------------------------------------------------------

    for soma_idx in range(
        S
    ):

        group = (

            "Coupled"

            if soma_idx < n_coupled

            else "Uncoupled"
        )


        d0 = (
            soma_idx
            * D
        )


        d1 = (
            (soma_idx + 1)
            * D
        )


        # ----------------------------------------------------
        # Input -> dendrite
        # ----------------------------------------------------

        grouped[
            group
        ][
            "Input → dendrite"
        ].append(

            remove_zeros(

                W_input[
                    :,
                    d0:d1
                ]
            )
        )


        # ----------------------------------------------------
        # Dendrite -> soma
        # ----------------------------------------------------

        grouped[
            group
        ][
            "Dendrite → soma"
        ].append(

            remove_zeros(

                W_soma[
                    d0:d1,
                    soma_idx
                ]
            )
        )


        # ----------------------------------------------------
        # Soma -> output
        # ----------------------------------------------------

        grouped[
            group
        ][
            "Soma → output"
        ].append(

            remove_zeros(

                W_output[
                    soma_idx,
                    :
                ]
            )
        )


    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    for group in [
        "Uncoupled",
        "Coupled",
    ]:

        for layer in LAYERS:

            arrays = [

                x

                for x in grouped[
                    group
                ][
                    layer
                ]

                if len(x)
            ]


            grouped[
                group
            ][
                layer
            ] = (

                np.concatenate(
                    arrays
                )

                if arrays

                else np.array([])
            )


    keras.backend.clear_session()


    return grouped


# ============================================================
# PRINT TRIAL AVAILABILITY
# ============================================================

print(
    "\n=============================================="
)

print(
    "DENDRITE-SWEEP TRIAL AVAILABILITY"
)

print(
    "=============================================="
)


for D in DENDS:

    lrf_trials = find_available_trials(
        LRF_FOLDER,
        D,
        SOMA,
    )


    c_trials = find_available_trials(
        C_FOLDER,
        D,
        SOMA,
    )


    print(
        f"D={D:2d} | "
        f"dANN-LRF n={len(lrf_trials):2d} {lrf_trials} | "
        f"dANN-C n={len(c_trials):2d} {c_trials}"
    )


# ============================================================
# EXTRACT SUMMARY DATA
#
# One row per:
#
# D
# trial
# model/group
# layer
# ============================================================

ROWS = []


for D in DENDS:

    # ========================================================
    # dANN-LRF
    # ========================================================

    lrf_trials = find_available_trials(
        LRF_FOLDER,
        D,
        SOMA,
    )


    for trial in lrf_trials:

        weights = load_lrf_weights(
            trial,
            D,
            SOMA,
        )


        for layer in LAYERS:

            stats = summarize_weights(

                weights[
                    layer
                ]
            )


            ROWS.append({

                "D":
                    D,

                "S":
                    SOMA,

                "trial":
                    trial,

                "model":
                    "dANN-LRF",

                "group":
                    "dANN-LRF",

                "layer":
                    layer,

                **stats,
            })


    # ========================================================
    # dANN-C 20%, λ=0.10
    # ========================================================

    c_trials = find_available_trials(
        C_FOLDER,
        D,
        SOMA,
    )


    for trial in c_trials:

        grouped = load_dannc_weights(
            trial,
            D,
            SOMA,
        )


        for group in [
            "Uncoupled",
            "Coupled",
        ]:

            for layer in LAYERS:

                stats = summarize_weights(

                    grouped[
                        group
                    ][
                        layer
                    ]
                )


                ROWS.append({

                    "D":
                        D,

                    "S":
                        SOMA,

                    "trial":
                        trial,

                    "model":
                        "dANN-C",

                    "group":
                        group,

                    "layer":
                        layer,

                    **stats,
                })


# ============================================================
# DATAFRAME
# ============================================================

DF = pd.DataFrame(
    ROWS
)


# ============================================================
# SAVE NUMERICAL DATA
# ============================================================

DF.to_csv(

    OUT
    / "weight_metrics_evolution_dendrites.csv",

    index=False,
)


# ============================================================
# METRICS
# ============================================================

METRICS = [

    (
        "fraction_positive",
        "Positive fraction",
        "Fraction positive",
    ),

    (
        "mean_positive",
        "Positive strength",
        "Mean positive weight",
    ),

    (
        "mean_negative",
        "Negative strength",
        "Mean negative weight",
    ),
]


# ============================================================
# DRAW ONE METRIC
# ============================================================

def plot_metric(
    ax,
    layer,
    metric,
    ylabel="",
):

    d = DF[
        DF[
            "layer"
        ] == layer
    ].copy()


    all_values = []


    # ========================================================
    # Three trajectories
    # ========================================================

    for group in [
        "dANN-LRF",
        "Uncoupled",
        "Coupled",
    ]:

        means = []
        sems = []


        for D in DENDS:

            vals = (

                d[
                    (
                        d[
                            "D"
                        ] == D
                    )

                    &

                    (
                        d[
                            "group"
                        ] == group
                    )
                ][
                    metric
                ]

                .dropna()

                .to_numpy()
            )


            mean, sem = mean_sem(
                vals
            )


            means.append(
                mean
            )


            sems.append(
                sem
            )


            all_values.extend(
                vals.tolist()
            )


        means = np.asarray(
            means,
            dtype=float,
        )


        sems = np.asarray(
            sems,
            dtype=float,
        )


        # ----------------------------------------------------
        # SEM
        # ----------------------------------------------------

        ax.fill_between(

            DENDS,

            means - sems,

            means + sems,

            color=
                COLORS[
                    group
                ],

            alpha=
                0.15,

            linewidth=
                0,
        )


        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        ax.plot(

            DENDS,

            means,

            marker=
                "o",

            markersize=
                5,

            linewidth=
                1.8,

            color=
                COLORS[
                    group
                ],

            label=
                group,
        )


    # ========================================================
    # X AXIS
    # ========================================================

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
        ylabel
    )


    zoom_axis(
        ax,
        all_values,
    )


# ============================================================
# FIGURE
#
# 3 rows:
#   Input -> dendrite
#   Dendrite -> soma
#   Soma -> output
#
# 3 cols:
#   Positive fraction
#   Positive strength
#   Negative strength
# ============================================================

fig, axes = plt.subplots(

    3,
    3,

    figsize=(
        11,
        9,
    ),
)


for row, layer in enumerate(
    LAYERS
):

    for col, (
        metric,
        title,
        ylabel,
    ) in enumerate(
        METRICS
    ):

        plot_metric(

            ax=
                axes[
                    row,
                    col
                ],

            layer=
                layer,

            metric=
                metric,

            ylabel=
                (
                    f"{layer}\n{ylabel}"

                    if col == 0

                    else ""
                ),
        )


        if row == 0:

            axes[
                row,
                col
            ].set_title(
                title
            )


        if row != 2:

            axes[
                row,
                col
            ].set_xlabel("")


# ============================================================
# LEGEND
# ============================================================

legend_handles = [

    Line2D(

        [0],
        [0],

        color=
            COLORS[
                "dANN-LRF"
            ],

        marker=
            "o",

        linewidth=
            1.8,

        label=
            "dANN-LRF",
    ),


    Line2D(

        [0],
        [0],

        color=
            COLORS[
                "Uncoupled"
            ],

        marker=
            "o",

        linewidth=
            1.8,

        label=
            "dANN-C uncoupled",
    ),


    Line2D(

        [0],
        [0],

        color=
            COLORS[
                "Coupled"
            ],

        marker=
            "o",

        linewidth=
            1.8,

        label=
            "dANN-C coupled",
    ),
]


axes[
    0,
    2
].legend(

    handles=
        legend_handles,

    frameon=
        False,

    fontsize=
        8,
)


sns.despine()

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

fig.savefig(

    OUT
    / "weight_metrics_evolution_dendrites.pdf",

    bbox_inches=
        "tight",
)


plt.close(
    fig
)