# weight_analysis_best_models.py

import pathlib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras

# Register custom dANN-C layer
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

WEIGHTS_FILE = (
    ROOT
    / "DATA"
    / "weights_best_models.pkl"
)

OUT = (
    ROOT
    / "FinalFigs_manuscript"
)

OUT.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TRIALS = [1, 2, 3, 4, 5]

SIGMA = 0.0

N_CLASSES = 10


# ============================================================
# MODELS
# ============================================================

MODELS = {

    "dANN-LRF": {
        "fraction": 0,
        "lambda": np.nan,
        "folder": None,
    },

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


LAYERS = [
    "Input → dendrite",
    "Dendrite → soma",
    "Soma → output",
]


# ============================================================
# COLORS
#
# Colors indicate population ONLY.
# ============================================================

GROUP_COLORS = {

    "dANN-LRF":
        "#ff9f9b",

    "Uncoupled":
        "#969696",

    "Coupled":
        "#8c2d8c",
}


# ============================================================
# STYLE
# ============================================================

sns.set_style("white")

plt.rcParams.update(
    my_style()
)


# ============================================================
# BASIC HELPERS
# ============================================================

def remove_zeros(weights):

    weights = np.asarray(
        weights,
        dtype=float,
    ).ravel()

    return weights[
        weights != 0
    ]



def summarize_weights(weights):

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

        "n_weights":
            len(weights),

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

        "mean_signed":
            np.mean(
                weights
            ),
    }



def mean_sem(values):

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


# ============================================================
# LOAD ORIGINAL dANN-LRF BEST MODEL
# ============================================================

print("\n==========================================")
print("LOADING ORIGINAL BEST dANN-LRF")
print("==========================================")


with open(
    WEIGHTS_FILE,
    "rb",
) as f:

    original_weights = pickle.load(f)


W_LRF = {

    "Input → dendrite":
        np.asarray(
            original_weights[
                "dendrites"
            ][
                "dend_ann_local_rfs"
            ]
        ),

    "Dendrite → soma":
        np.asarray(
            original_weights[
                "soma"
            ][
                "dend_ann_local_rfs"
            ]
        ),

    "Soma → output":
        np.asarray(
            original_weights[
                "output"
            ][
                "dend_ann_local_rfs"
            ]
        ),
}


for layer in LAYERS:

    print(
        f"{layer}: "
        f"{W_LRF[layer].shape}"
    )


# ============================================================
# INFER dANN-LRF ARCHITECTURE
# ============================================================

n_output_weights = (
    W_LRF[
        "Soma → output"
    ].shape[0]
)


if (
    n_output_weights
    % N_CLASSES
    != 0
):

    raise ValueError(
        "Cannot infer S from dANN-LRF output weights."
    )


BEST_S = (
    n_output_weights
    // N_CLASSES
)


n_soma_weights = (
    W_LRF[
        "Dendrite → soma"
    ].shape[0]
)


if (
    n_soma_weights
    % BEST_S
    != 0
):

    raise ValueError(
        "Cannot infer D from dANN-LRF soma weights."
    )


BEST_D = (
    n_soma_weights
    // BEST_S
)


print("\n==========================================")
print("BEST dANN-LRF ARCHITECTURE")
print("==========================================")

print(
    f"D = {BEST_D}"
)

print(
    f"S = {BEST_S}"
)


# ============================================================
# VERIFY ARCHITECTURE
# ============================================================

expected_input = (
    16
    * BEST_D
    * BEST_S
)


actual_input = (
    W_LRF[
        "Input → dendrite"
    ].shape[0]
)


print(
    f"\nExpected input weights: {expected_input}"
)

print(
    f"Stored input weights:   {actual_input}"
)


if (
    expected_input
    != actual_input
):

    raise ValueError(
        "dANN-LRF architecture inference failed."
    )


for layer in LAYERS:

    if (
        W_LRF[layer].shape[1]
        != len(TRIALS)
    ):

        raise ValueError(
            f"Unexpected number of trials for {layer}."
        )


print(
    "\n✓ dANN-LRF architecture internally consistent."
)


# ============================================================
# FIND MATCHED dANN-C MODEL
# ============================================================

def find_model_file(
    model_name,
    trial,
):

    folder = (
        MODELS[
            model_name
        ]["folder"]
    )


    fname = folder / (

        f"model_sigma_{SIGMA}"
        f"_trial_{trial}"
        f"_dends_{BEST_D}"
        f"_soma_{BEST_S}.keras"
    )


    if not fname.exists():

        raise FileNotFoundError(

            "\nMissing matched model:\n"
            f"{fname}\n\n"
            f"Required architecture: "
            f"D={BEST_D}, S={BEST_S}"
        )


    return fname


# ============================================================
# EXTRACT COUPLED AND UNCOUPLED WEIGHTS
# ============================================================

def get_grouped_model_weights(
    model_name,
    trial,
):
    """
    Split all three weight populations according to whether
    they belong to a coupled or uncoupled soma.
    """

    fname = find_model_file(
        model_name,
        trial,
    )


    model = keras.models.load_model(
        fname
    )


    fraction = (
        MODELS[
            model_name
        ]["fraction"]
        / 100
    )


    n_coupled = int(
        round(
            BEST_S
            * fraction
        )
    )


    grouped = {

        layer: {
            "Coupled": [],
            "Uncoupled": [],
        }

        for layer in LAYERS
    }


    # ========================================================
    # MATRICES
    # ========================================================

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


    # ========================================================
    # ASSIGN EACH WEIGHT TO ITS SOMA
    # ========================================================

    for soma_idx in range(
        BEST_S
    ):

        group = (

            "Coupled"

            if soma_idx < n_coupled

            else "Uncoupled"
        )


        d0 = (
            soma_idx
            * BEST_D
        )


        d1 = (
            (soma_idx + 1)
            * BEST_D
        )


        # ----------------------------------------------------
        # Input -> dendrite
        # ----------------------------------------------------

        grouped[
            "Input → dendrite"
        ][group].append(

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
            "Dendrite → soma"
        ][group].append(

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
            "Soma → output"
        ][group].append(

            remove_zeros(

                W_output[
                    soma_idx,
                    :
                ]
            )
        )


    # ========================================================
    # CONCATENATE SOMATA
    # ========================================================

    for layer in LAYERS:

        for group in [
            "Coupled",
            "Uncoupled",
        ]:

            arrays = [

                x

                for x in grouped[
                    layer
                ][
                    group
                ]

                if len(x)
            ]


            grouped[
                layer
            ][
                group
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
# CHECK MODEL AVAILABILITY
# ============================================================

print("\n==========================================")
print("MATCHED MODEL AVAILABILITY")
print("==========================================")


print(
    f"dANN-LRF: D={BEST_D}, S={BEST_S}"
)


for model_name in MODELS:

    if model_name == "dANN-LRF":
        continue


    n_found = 0


    for trial in TRIALS:

        try:

            find_model_file(
                model_name,
                trial,
            )

            n_found += 1

        except FileNotFoundError:

            pass


    print(
        f"{model_name:25s}: "
        f"{n_found}/5"
    )


    if n_found != 5:

        raise RuntimeError(

            f"{model_name} incomplete at "
            f"D={BEST_D}, S={BEST_S}."
        )


# ============================================================
# STORAGE
# ============================================================

GROUPED_RAW = {}

grouped_summary_rows = []


# ============================================================
# ADD dANN-LRF
# ============================================================

GROUPED_RAW[
    "dANN-LRF"
] = {

    layer: []

    for layer in LAYERS
}


print("\n==========================================")
print("ADDING dANN-LRF")
print("==========================================")


for trial_idx, trial in enumerate(
    TRIALS
):

    print(
        f"trial {trial}"
    )


    for layer in LAYERS:

        weights = remove_zeros(

            W_LRF[
                layer
            ][
                :,
                trial_idx
            ]
        )


        GROUPED_RAW[
            "dANN-LRF"
        ][
            layer
        ].append(
            weights
        )


        stats = summarize_weights(
            weights
        )


        row = {

            "model":
                "dANN-LRF",

            "trial":
                trial,

            "D":
                BEST_D,

            "S":
                BEST_S,

            "fraction":
                0,

            "lambda":
                np.nan,

            "layer":
                layer,

            "group":
                "dANN-LRF",
        }


        row.update(
            stats
        )


        grouped_summary_rows.append(
            row
        )


# ============================================================
# ADD dANN-C GROUPED WEIGHTS
# ============================================================

for model_name in MODELS:

    if model_name == "dANN-LRF":
        continue


    GROUPED_RAW[
        model_name
    ] = {

        layer: {
            "Coupled": [],
            "Uncoupled": [],
        }

        for layer in LAYERS
    }


    print(
        f"\n{model_name}"
    )


    for trial in TRIALS:

        print(
            f"  trial {trial}"
        )


        grouped = (
            get_grouped_model_weights(
                model_name,
                trial,
            )
        )


        for layer in LAYERS:

            for group in [
                "Uncoupled",
                "Coupled",
            ]:

                weights = grouped[
                    layer
                ][
                    group
                ]


                GROUPED_RAW[
                    model_name
                ][
                    layer
                ][
                    group
                ].append(
                    weights
                )


                stats = summarize_weights(
                    weights
                )


                row = {

                    "model":
                        model_name,

                    "trial":
                        trial,

                    "D":
                        BEST_D,

                    "S":
                        BEST_S,

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

                    "group":
                        group,
                }


                row.update(
                    stats
                )


                grouped_summary_rows.append(
                    row
                )


# ============================================================
# CONCATENATE RAW WEIGHTS ACROSS TRIALS
# ============================================================

for layer in LAYERS:

    GROUPED_RAW[
        "dANN-LRF"
    ][
        layer
    ] = np.concatenate(

        GROUPED_RAW[
            "dANN-LRF"
        ][
            layer
        ]
    )


for model_name in MODELS:

    if model_name == "dANN-LRF":
        continue


    for layer in LAYERS:

        for group in [
            "Uncoupled",
            "Coupled",
        ]:

            GROUPED_RAW[
                model_name
            ][
                layer
            ][
                group
            ] = np.concatenate(

                GROUPED_RAW[
                    model_name
                ][
                    layer
                ][
                    group
                ]
            )


GROUPED_SUMMARY = pd.DataFrame(
    grouped_summary_rows
)


# ============================================================
# SAVE NUMERICAL DATA
# ============================================================

SUMMARY_FILE = (
    OUT
    / "weights_best_models_grouped_summary.csv"
)


RAW_FILE = (
    OUT
    / "weights_best_models_grouped_raw.pkl"
)


GROUPED_SUMMARY.to_csv(
    SUMMARY_FILE,
    index=False,
)


with open(
    RAW_FILE,
    "wb",
) as f:

    pickle.dump(

        {
            "weights":
                GROUPED_RAW,

            "D":
                BEST_D,

            "S":
                BEST_S,
        },

        f,

        protocol=
            pickle.HIGHEST_PROTOCOL,
    )


# ============================================================
# PRINT GROUPED SUMMARY
# ============================================================

print("\n==========================================")
print("GROUPED TRIAL-LEVEL SUMMARY")
print("==========================================")


print(

    GROUPED_SUMMARY.groupby(
        [
            "model",
            "layer",
            "group",
        ]
    )[
        [
            "fraction_positive",
            "fraction_negative",
            "mean_positive",
            "mean_negative",
        ]
    ]

    .agg(
        [
            "mean",
            "std",
            "median",
        ]
    )

    .round(4)
)


# ============================================================
# Y-AXIS ZOOM HELPER
# ============================================================

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


    if not len(values):

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
# LINE PLOT: COUPLING FRACTION
#
# x = 10, 20, 30, 40 %
#
# grey:
# uncoupled
#
# purple:
# coupled
#
# pink dashed horizontal line:
# dANN-LRF mean
#
# pink shaded area:
# dANN-LRF ± SEM
# ============================================================

def plot_fraction_evolution(
    ax,
    layer,
    metric,
    ylabel="",
):

    data = GROUPED_SUMMARY[
        GROUPED_SUMMARY[
            "layer"
        ] == layer
    ].copy()


    fractions = [
        10,
        20,
        30,
        40,
    ]


    # ========================================================
    # dANN-LRF REFERENCE
    # ========================================================

    lrf_values = (

        data[
            data[
                "model"
            ] == "dANN-LRF"
        ][
            metric
        ]

        .dropna()

        .to_numpy()
    )


    lrf_mean, lrf_sem = mean_sem(
        lrf_values
    )


    ax.axhline(

        lrf_mean,

        color=
            GROUP_COLORS[
                "dANN-LRF"
            ],

        linestyle=
            "--",

        linewidth=
            1.5,

        label=
            "dANN-LRF",
    )


    if np.isfinite(
        lrf_sem
    ):

        ax.axhspan(

            lrf_mean
            - lrf_sem,

            lrf_mean
            + lrf_sem,

            color=
                GROUP_COLORS[
                    "dANN-LRF"
                ],

            alpha=
                0.12,

            linewidth=
                0,
        )


    # ========================================================
    # COUPLED / UNCOUPLED
    # ========================================================

    all_values = list(
        lrf_values
    )


    for group in [
        "Uncoupled",
        "Coupled",
    ]:

        means = []

        sems = []


        for fraction in fractions:

            values = (

                data[
                    (
                        data[
                            "fraction"
                        ] == fraction
                    )

                    &

                    (
                        data[
                            "lambda"
                        ] == 0.10
                    )

                    &

                    (
                        data[
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
                values
            )


            means.append(
                mean
            )

            sems.append(
                sem
            )


            all_values.extend(
                values.tolist()
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
        # SEM shaded region
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
        fractions
    )


    ax.set_xticklabels(
        [
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
# LINE PLOT: LAMBDA
#
# x = .05, .10, .20
#
# grey:
# uncoupled
#
# purple:
# coupled
#
# pink:
# dANN-LRF reference mean ± SEM
# ============================================================

def plot_lambda_evolution(
    ax,
    layer,
    metric,
    ylabel="",
):

    data = GROUPED_SUMMARY[
        GROUPED_SUMMARY[
            "layer"
        ] == layer
    ].copy()


    lambdas = [
        0.05,
        0.10,
        0.20,
    ]


    # ========================================================
    # dANN-LRF
    # ========================================================

    lrf_values = (

        data[
            data[
                "model"
            ] == "dANN-LRF"
        ][
            metric
        ]

        .dropna()

        .to_numpy()
    )


    lrf_mean, lrf_sem = mean_sem(
        lrf_values
    )


    ax.axhline(

        lrf_mean,

        color=
            GROUP_COLORS[
                "dANN-LRF"
            ],

        linestyle=
            "--",

        linewidth=
            1.5,

        label=
            "dANN-LRF",
    )


    if np.isfinite(
        lrf_sem
    ):

        ax.axhspan(

            lrf_mean
            - lrf_sem,

            lrf_mean
            + lrf_sem,

            color=
                GROUP_COLORS[
                    "dANN-LRF"
                ],

            alpha=
                0.12,

            linewidth=
                0,
        )


    all_values = list(
        lrf_values
    )


    # ========================================================
    # COUPLED / UNCOUPLED
    # ========================================================

    for group in [
        "Uncoupled",
        "Coupled",
    ]:

        means = []

        sems = []


        for lam in lambdas:

            values = (

                data[
                    (
                        data[
                            "lambda"
                        ] == lam
                    )

                    &

                    (
                        data[
                            "fraction"
                        ] == 20
                    )

                    &

                    (
                        data[
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
                values
            )


            means.append(
                mean
            )

            sems.append(
                sem
            )


            all_values.extend(
                values.tolist()
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
        # SEM shading
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
        lambdas
    )


    ax.set_xticklabels(
        [
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
# REMOVE DUPLICATE LEGENDS
# ============================================================

def clean_legends(
    axes,
):

    for row in range(
        axes.shape[0]
    ):

        for col in range(
            axes.shape[1]
        ):

            legend = axes[
                row,
                col
            ].get_legend()


            if legend is not None:

                legend.remove()


    handles, labels = (
        axes[
            0,
            -1
        ].get_legend_handles_labels()
    )


    axes[
        0,
        -1
    ].legend(

        handles,
        labels,

        frameon=
            False,

        fontsize=
            8,
    )


# ============================================================
# GENERIC COUPLING-FRACTION FIGURE
# ============================================================

def make_fraction_line_figure(
    metric_left,
    title_left,
    ylabel_left,
    metric_right,
    title_right,
    filename,
):

    fig, axes = plt.subplots(

        3,
        2,

        figsize=(
            8,
            8,
        ),
    )


    for row, layer in enumerate(
        LAYERS
    ):

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        plot_fraction_evolution(

            ax=
                axes[
                    row,
                    0
                ],

            layer=
                layer,

            metric=
                metric_left,

            ylabel=
                ylabel_left,
        )


        axes[
            row,
            0
        ].set_title(

            f"{layer}\n"
            f"{title_left}"
        )


        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        plot_fraction_evolution(

            ax=
                axes[
                    row,
                    1
                ],

            layer=
                layer,

            metric=
                metric_right,

            ylabel="",
        )


        axes[
            row,
            1
        ].set_title(

            f"{layer}\n"
            f"{title_right}"
        )


        if row != 2:

            axes[
                row,
                0
            ].set_xlabel("")

            axes[
                row,
                1
            ].set_xlabel("")


    clean_legends(
        axes
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
# GENERIC LAMBDA FIGURE
# ============================================================

def make_lambda_line_figure(
    metric_left,
    title_left,
    ylabel_left,
    metric_right,
    title_right,
    filename,
):

    fig, axes = plt.subplots(

        3,
        2,

        figsize=(
            8,
            8,
        ),
    )


    for row, layer in enumerate(
        LAYERS
    ):

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        plot_lambda_evolution(

            ax=
                axes[
                    row,
                    0
                ],

            layer=
                layer,

            metric=
                metric_left,

            ylabel=
                ylabel_left,
        )


        axes[
            row,
            0
        ].set_title(

            f"{layer}\n"
            f"{title_left}"
        )


        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        plot_lambda_evolution(

            ax=
                axes[
                    row,
                    1
                ],

            layer=
                layer,

            metric=
                metric_right,

            ylabel="",
        )


        axes[
            row,
            1
        ].set_title(

            f"{layer}\n"
            f"{title_right}"
        )


        if row != 2:

            axes[
                row,
                0
            ].set_xlabel("")

            axes[
                row,
                1
            ].set_xlabel("")


    clean_legends(
        axes
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
# FIGURE 1
#
# FRACTION POSITIVE / NEGATIVE
# vs COUPLING FRACTION
# ============================================================

make_fraction_line_figure(

    metric_left=
        "fraction_positive",

    title_left=
        "Positive weights",

    ylabel_left=
        "Fraction of weights",

    metric_right=
        "fraction_negative",

    title_right=
        "Negative weights",

    filename=
        "weights_best_models_fraction_coupling.pdf",
)


# ============================================================
# FIGURE 2
#
# FRACTION POSITIVE / NEGATIVE
# vs LAMBDA
# ============================================================

make_lambda_line_figure(

    metric_left=
        "fraction_positive",

    title_left=
        "Positive weights",

    ylabel_left=
        "Fraction of weights",

    metric_right=
        "fraction_negative",

    title_right=
        "Negative weights",

    filename=
        "weights_best_models_fraction_lambda.pdf",
)


# ============================================================
# FIGURE 3
#
# POSITIVE / NEGATIVE STRENGTH
# vs COUPLING FRACTION
# ============================================================

make_fraction_line_figure(

    metric_left=
        "mean_positive",

    title_left=
        "Positive weight strength",

    ylabel_left=
        "Mean weight",

    metric_right=
        "mean_negative",

    title_right=
        "Negative weight strength",

    filename=
        "weights_best_models_strength_coupling.pdf",
)


# ============================================================
# FIGURE 4
#
# POSITIVE / NEGATIVE STRENGTH
# vs LAMBDA
# ============================================================

make_lambda_line_figure(

    metric_left=
        "mean_positive",

    title_left=
        "Positive weight strength",

    ylabel_left=
        "Mean weight",

    metric_right=
        "mean_negative",

    title_right=
        "Negative weight strength",

    filename=
        "weights_best_models_strength_lambda.pdf",
)


# ============================================================
# KDE LEGEND HELPER
# ============================================================

def add_group_legend(
    ax,
):

    from matplotlib.lines import Line2D


    handles = [

        Line2D(
            [0],
            [0],
            color=
                GROUP_COLORS[
                    "dANN-LRF"
                ],
            lw=2,
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
            lw=2,
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
            lw=2,
            label=
                "Coupled",
        ),
    ]


    ax.legend(

        handles=
            handles,

        frameon=
            False,

        fontsize=
            8,
    )


# ============================================================
# FIGURE 5
#
# WEIGHT DISTRIBUTIONS
# COUPLING FRACTION
#
# dANN-LRF | 10 | 20 | 30 | 40
#
# dANN-C panels:
# grey   = uncoupled
# purple = coupled
# ============================================================

fig, axes = plt.subplots(

    3,
    5,

    figsize=(
        15,
        7.5,
    ),

    sharey="row",
)


# ------------------------------------------------------------
# FIRST COLUMN = dANN-LRF
# ------------------------------------------------------------

for row, layer in enumerate(
    LAYERS
):

    ax = axes[
        row,
        0
    ]


    sns.kdeplot(

        x=
            GROUPED_RAW[
                "dANN-LRF"
            ][
                layer
            ],

        ax=
            ax,

        color=
            GROUP_COLORS[
                "dANN-LRF"
            ],

        linewidth=
            1.8,

        fill=
            False,

        common_norm=
            False,
    )


    ax.axvline(

        0,

        color=
            "0.7",

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


    if row == 2:

        ax.set_xlabel(
            "Weight"
        )

    else:

        ax.set_xlabel("")


# ------------------------------------------------------------
# FOLLOWING COLUMNS = COUPLING FRACTION
# ------------------------------------------------------------

for col, model_name in enumerate(
    COUPLING_MODELS,
    start=1,
):

    fraction = (
        MODELS[
            model_name
        ]["fraction"]
    )


    for row, layer in enumerate(
        LAYERS
    ):

        ax = axes[
            row,
            col
        ]


        for group in [
            "Uncoupled",
            "Coupled",
        ]:

            sns.kdeplot(

                x=
                    GROUPED_RAW[
                        model_name
                    ][
                        layer
                    ][
                        group
                    ],

                ax=
                    ax,

                color=
                    GROUP_COLORS[
                        group
                    ],

                linewidth=
                    1.8,

                fill=
                    False,

                common_norm=
                    False,
            )


        ax.axvline(

            0,

            color=
                "0.7",

            ls=
                "--",

            lw=
                0.8,
        )


        if row == 0:

            ax.set_title(
                f"{fraction}%"
            )


        ax.set_ylabel("")


        if row == 2:

            ax.set_xlabel(
                "Weight"
            )

        else:

            ax.set_xlabel("")


add_group_legend(
    axes[
        0,
        4
    ]
)


sns.despine()

fig.tight_layout()


fig.savefig(

    OUT
    / "weights_best_models_LRF_vs_uncoupled_vs_coupled_fraction.pdf",

    bbox_inches=
        "tight",
)


plt.close(
    fig
)


# ============================================================
# FIGURE 6
#
# WEIGHT DISTRIBUTIONS
# LAMBDA
#
# dANN-LRF | .05 | .10 | .20
# ============================================================

fig, axes = plt.subplots(

    3,
    4,

    figsize=(
        12,
        7.5,
    ),

    sharey="row",
)


# ------------------------------------------------------------
# FIRST COLUMN = dANN-LRF
# ------------------------------------------------------------

for row, layer in enumerate(
    LAYERS
):

    ax = axes[
        row,
        0
    ]


    sns.kdeplot(

        x=
            GROUPED_RAW[
                "dANN-LRF"
            ][
                layer
            ],

        ax=
            ax,

        color=
            GROUP_COLORS[
                "dANN-LRF"
            ],

        linewidth=
            1.8,

        fill=
            False,

        common_norm=
            False,
    )


    ax.axvline(

        0,

        color=
            "0.7",

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


    if row == 2:

        ax.set_xlabel(
            "Weight"
        )

    else:

        ax.set_xlabel("")


# ------------------------------------------------------------
# FOLLOWING COLUMNS = LAMBDA
# ------------------------------------------------------------

for col, model_name in enumerate(
    LAMBDA_MODELS,
    start=1,
):

    lam = (
        MODELS[
            model_name
        ]["lambda"]
    )


    for row, layer in enumerate(
        LAYERS
    ):

        ax = axes[
            row,
            col
        ]


        for group in [
            "Uncoupled",
            "Coupled",
        ]:

            sns.kdeplot(

                x=
                    GROUPED_RAW[
                        model_name
                    ][
                        layer
                    ][
                        group
                    ],

                ax=
                    ax,

                color=
                    GROUP_COLORS[
                        group
                    ],

                linewidth=
                    1.8,

                fill=
                    False,

                common_norm=
                    False,
            )


        ax.axvline(

            0,

            color=
                "0.7",

            ls=
                "--",

            lw=
                0.8,
        )


        if row == 0:

            ax.set_title(
                f"λ = {lam}"
            )


        ax.set_ylabel("")


        if row == 2:

            ax.set_xlabel(
                "Weight"
            )

        else:

            ax.set_xlabel("")


add_group_legend(
    axes[
        0,
        3
    ]
)


sns.despine()

fig.tight_layout()


fig.savefig(

    OUT
    / "weights_best_models_LRF_vs_uncoupled_vs_coupled_lambda.pdf",

    bbox_inches=
        "tight",
)


plt.close(
    fig
)


# ============================================================
# DONE
# ============================================================

print("\n==========================================")
print("BEST-MODEL WEIGHT ANALYSIS COMPLETE")
print("==========================================")


print(
    f"\nMatched architecture:"
    f"\nD = {BEST_D}"
    f"\nS = {BEST_S}"
)


print(
    "\nColors:"
)

print(
    "dANN-LRF  = pink reference"
)

print(
    "Uncoupled = grey"
)

print(
    "Coupled   = purple"
)


print(
    "\nLine plots:"
)

print(
    "solid line = mean across N=5 trials"
)

print(
    "shaded region = SEM"
)

print(
    "pink dashed line/shading = dANN-LRF mean ± SEM"
)


print(
    "\nSaved:"
)


for fname in [

    "weights_best_models_fraction_coupling.pdf",

    "weights_best_models_fraction_lambda.pdf",

    "weights_best_models_strength_coupling.pdf",

    "weights_best_models_strength_lambda.pdf",

    "weights_best_models_LRF_vs_uncoupled_vs_coupled_fraction.pdf",

    "weights_best_models_LRF_vs_uncoupled_vs_coupled_lambda.pdf",

    "weights_best_models_grouped_summary.csv",

    "weights_best_models_grouped_raw.pkl",
]:

    print(
        OUT
        / fname
    )