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


def num_trainable_params(D, S, n_syn=16, n_class=10):
    return (n_syn + 1) * D * S + (D + 1) * S + (S + 1) * n_class


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = pathlib.Path.home() / "dendritic_anns"

OLD_FILE = ROOT / "DATA/results_fmnist_1_layer/output_all_final.pkl"
C_DIR = ROOT / "DATA_C/results_fmnist_1_layer/dend_ann_local_rfs_C"

SAVE = ROOT / "FinalFigs_manuscript/figure_2_models_2_6_9_13.pdf"
SAVE.parent.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# ORIGINAL PRECOMPUTED MODELS
# ---------------------------------------------------------------------

with open(OLD_FILE, "rb") as f:
    results = pickle.load(f)

df_old = fix_names(results["testing"]).copy()
df_old["test_acc"] *= 100

models_old = ["dANN-LRF", "vANN-LRF", "sANN-LRF"]
df_old = df_old[df_old["model"].isin(models_old)].copy()


# Keep original vANN purely as Figure-2 reference line
df_vann = fix_names(results["testing"]).copy()
df_vann["test_acc"] *= 100
vann_metrics = calculate_best_model(df_vann[df_vann["model"] == "vANN"])

vann_min_loss = np.nanmin(vann_metrics[:, 0])
vann_max_acc = np.nanmax(vann_metrics[:, 1])


# ---------------------------------------------------------------------
# dANN-C RESULTS
# ---------------------------------------------------------------------

rows = []

pattern = re.compile(
    r"results_sigma_0\.0_trial_(\d+)_dends_(\d+)_soma_(\d+)\.pkl"
)

for file in C_DIR.glob("results_sigma_0.0_trial_*_dends_*_soma_*.pkl"):

    m = pattern.fullmatch(file.name)
    if not m:
        continue

    trial, dends, soma = map(int, m.groups())

    with open(file, "rb") as f:
        out = pickle.load(f)

    rows.append({
        "model": "dANN-C",
        "trial": trial,
        "num_dends": dends,
        "num_soma": soma,
        "test_loss": float(np.asarray(out["test_loss"]).squeeze()),
        "test_acc": float(np.asarray(out["test_acc"]).squeeze()) * 100,
        "trainable_params": num_trainable_params(dends, soma),
    })

df_C = pd.DataFrame(rows)

print(f"All dANN-C runs found: {len(df_C)}")

print("\nRuns per D × S before filtering:")
print(
    df_C.groupby(["num_dends", "num_soma"])
    .size()
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# FIGURE 2 MUST MATCH THE ORIGINAL N = 5 DESIGN
# Keep only trials 1–5.
#
# Trials 6–15 were added later only for the D=8, S=512
# architecture and should not enter the Figure 2 grid.
# ------------------------------------------------------------

df_C = df_C[df_C["trial"].between(1, 5)].copy()

print(f"\ndANN-C runs used for Figure 2: {len(df_C)}")

counts = (
    df_C.groupby(["num_dends", "num_soma"])
    .size()
)

print("\nRuns per D × S after filtering:")
print(counts.value_counts().sort_index())

# Sanity checks
expected_architectures = 6 * 5       # 6 D values × 5 S values
expected_trials = 5
expected_runs = expected_architectures * expected_trials

assert len(counts) == expected_architectures, (
    f"Expected {expected_architectures} D×S architectures, "
    f"found {len(counts)}"
)

assert (counts == expected_trials).all(), (
    "Some D×S architectures do not contain exactly 5 trials:\n"
    f"{counts[counts != expected_trials]}"
)

assert len(df_C) == expected_runs, (
    f"Expected {expected_runs} dANN-C runs after filtering, "
    f"found {len(df_C)}"
)


# ---------------------------------------------------------------------
# COMBINE
# ---------------------------------------------------------------------

df = pd.concat([df_old, df_C], ignore_index=True)

order = ["dANN-LRF", "dANN-C", "sANN-LRF", "vANN-LRF"]

palette = {
    "dANN-LRF": "#ff9f9b",
    "dANN-C": "#7f0000",
    "sANN-LRF": "#8de5a1",
    "vANN-LRF": "#a1c9f4",
}


# ---------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------

sns.set_style("white")
fig, ax = plt.subplots(2, 2, figsize=(7, 7))

# B — test loss vs parameters
sns.lineplot(
    data=df,
    x="trainable_params",
    y="test_loss",
    hue="model",
    hue_order=order,
    style="model",
    markers=True,
    dashes=False,
    palette=palette,
    errorbar=("ci", 95),
    ax=ax[0, 0],
)

ax[0, 0].axhline(vann_min_loss, ls="--", c="k", lw=1)
ax[0, 0].set_xscale("log")
ax[0, 0].set_xlabel("trainable parameters")
ax[0, 0].set_ylabel("test loss")
ax[0, 0].legend_.remove()


# C — accuracy vs parameters
sns.lineplot(
    data=df,
    x="trainable_params",
    y="test_acc",
    hue="model",
    hue_order=order,
    style="model",
    markers=True,
    dashes=False,
    palette=palette,
    errorbar=("ci", 95),
    ax=ax[0, 1],
)

ax[0, 1].axhline(vann_max_acc, ls="--", c="k", lw=1)
ax[0, 1].set_xscale("log")
ax[0, 1].set_xlabel("trainable parameters")
ax[0, 1].set_ylabel("test accuracy (%)")


# D/E — same logic as original Figure 2:
# only S = 256 and 512
df_ds = df[
    (df["num_soma"] > 200) &
    (df["model"] != "vANN-LRF")
].copy()


# D — test loss vs dendrites
sns.lineplot(
    data=df_ds,
    x="num_dends",
    y="test_loss",
    hue="model",
    hue_order=order,
    style="num_soma",
    markers=True,
    palette=palette,
    errorbar=("ci", 95),
    ax=ax[1, 0],
)

ax[1, 0].axhline(vann_min_loss, ls="--", c="k", lw=1)
ax[1, 0].set_xscale("log", base=2)
ax[1, 0].set_xlabel("dendrites per soma")
ax[1, 0].set_ylabel("test loss")
ax[1, 0].legend_.remove()


# E — accuracy vs dendrites
sns.lineplot(
    data=df_ds,
    x="num_dends",
    y="test_acc",
    hue="model",
    hue_order=order,
    style="num_soma",
    markers=True,
    palette=palette,
    errorbar=("ci", 95),
    ax=ax[1, 1],
)

ax[1, 1].axhline(vann_max_acc, ls="--", c="k", lw=1)
ax[1, 1].set_xscale("log", base=2)
ax[1, 1].set_xlabel("dendrites per soma")
ax[1, 1].set_ylabel("test accuracy (%)")


for a, label in zip(ax.flat, ["b", "c", "d", "e"]):
    a.text(
        -0.15, 1.05, label,
        transform=a.transAxes,
        fontsize=13,
        fontweight="bold",
    )

sns.despine()
plt.tight_layout()
plt.savefig(SAVE, dpi=600, bbox_inches="tight")
plt.savefig(SAVE.with_suffix(".svg"), bbox_inches="tight")

print(f"\nSaved: {SAVE}")