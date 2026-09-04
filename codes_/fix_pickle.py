#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append sign-constrained (Dale's-law style) runs to an existing aggregated result set.

Motivation
----------
The aggregated file `output_all_final.pkl` was built from raw per-run pickles that
are no longer present on disk. Re-running the aggregation script would therefore
rebuild the dataframe from whatever raw folders currently exist, which is only the
new sign-constrained model, and would silently discard the twelve models already in
the file. The alternative, re-running the full 140-configuration sweep, costs days
on CPU.

This script avoids both. It reads the raw pickles written by `main.py` for the new
model, reconstructs exactly the columns the aggregated dataframe uses, concatenates
them onto the existing `testing` dataframe, and writes a new aggregated file under a
different name. The original is never modified.

What it measures
----------------
For every raw run pickle it extracts test accuracy and test loss, parses the run
configuration (sigma, trial, dendrites, soma) out of the filename, and computes
`trainable_params` as the number of structurally connected weights, that is the
count of nonzero entries across all mask arrays saved with the run. This
reproduces the convention in the existing file, verified against the dend_ann_random
row with dends=1 and soma=32, which gives 938 by both routes.

`trainable_params_grouped` is not recomputed from first principles, since its
binning rule lives in the original aggregation script. It is inherited from the
nearest existing `trainable_params` value in the reference dataframe. Neither that
column nor `num_epochs_min` is used by figure_2.py, so an approximation there is
harmless; both are filled only to keep the schema consistent.

Usage
-----
    # from the repository root
    python append_sign_constrained.py \
        --reference DATA/results_fmnist_1_layer/output_all_final.pkl \
        --raw-dir codes_/DATA/results_fmnist_1_layer/dend_ann_random_sign_constrained \
        --model-name dend_ann_random_sign_constrained \
        --output DATA/results_fmnist_1_layer/output_all_with_dl.pkl

    # inspect what would be appended without writing anything
    python append_sign_constrained.py --reference ... --raw-dir ... --dry-run
"""

import re
import sys
import pickle
import pathlib
import argparse

import numpy as np
import pandas as pd


FNAME_RE = re.compile(
    r"^results_sigma_(?P<sigma>[0-9.]+)"
    r"_trial_(?P<trial>\d+)"
    r"_dends_(?P<dends>\d+)"
    r"_soma_(?P<soma>\d+)\.pkl$"
)


def scalar(value):
    """Return a float from either a scalar or a one-element/last-epoch sequence."""
    arr = np.asarray(value).squeeze()
    if arr.ndim == 0:
        return float(arr)
    return float(arr[-1])


def count_trainable(masks):
    """Number of structurally connected weights across all mask arrays."""
    total = 0
    for m in masks:
        if m is None:
            continue
        total += int(np.count_nonzero(np.asarray(m)))
    return total


def epochs_to_min_val_loss(out):
    """1-based epoch index at which validation loss was lowest."""
    val_loss = out.get("val_loss")
    if val_loss is None:
        return np.nan
    return int(np.argmin(np.asarray(val_loss))) + 1


def nearest_group(reference_df, n_params):
    """Inherit trainable_params_grouped from the closest existing run size."""
    if "trainable_params_grouped" not in reference_df.columns:
        return np.nan
    ref = reference_df[["trainable_params", "trainable_params_grouped"]].dropna()
    if ref.empty:
        return np.nan
    idx = (ref["trainable_params"] - n_params).abs().idxmin()
    return ref.loc[idx, "trainable_params_grouped"]


def build_rows(raw_dir, model_name, reference_df):
    rows = []
    skipped = []
    for path in sorted(pathlib.Path(raw_dir).glob("results_*.pkl")):
        match = FNAME_RE.match(path.name)
        if match is None:
            skipped.append(path.name)
            continue
        with open(path, "rb") as handle:
            out = pickle.load(handle)

        masks = out.get("Masks")
        if masks is None:
            skipped.append(f"{path.name} (no Masks)")
            continue

        n_params = count_trainable(masks)
        rows.append({
            "test_acc": scalar(out["test_acc"]),
            "test_loss": scalar(out["test_loss"]),
            "model": model_name,
            "num_dends": int(match.group("dends")),
            "num_soma": int(match.group("soma")),
            "sigma": float(match.group("sigma")),
            "trial": int(match.group("trial")),
            "trainable_params": n_params,
            "trainable_params_grouped": nearest_group(reference_df, n_params),
            "num_epochs_min": epochs_to_min_val_loss(out),
        })
    return pd.DataFrame(rows), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True,
                        help="existing aggregated pickle to append to")
    parser.add_argument("--raw-dir", required=True,
                        help="directory of per-run result pickles for the new model")
    parser.add_argument("--model-name", default="dend_ann_random_sign_constrained",
                        help="raw model name, must match the fix_names mapping key")
    parser.add_argument("--output", default=None,
                        help="where to write the merged pickle")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be appended, write nothing")
    args = parser.parse_args()

    with open(args.reference, "rb") as handle:
        results = pickle.load(handle)

    reference_df = results["testing"]
    new_df, skipped = build_rows(args.raw_dir, args.model_name, reference_df)

    if new_df.empty:
        sys.exit(f"no parsable run pickles found in {args.raw_dir}")

    missing = set(reference_df.columns) - set(new_df.columns)
    extra = set(new_df.columns) - set(reference_df.columns)
    if missing or extra:
        sys.exit(f"schema mismatch. missing={sorted(missing)} extra={sorted(extra)}")

    if args.model_name in set(reference_df["model"].unique()):
        sys.exit(f"{args.model_name} is already present in the reference file")

    print(f"parsed {len(new_df)} runs from {args.raw_dir}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    print(new_df.groupby(["num_dends", "num_soma"]).size().to_string())
    print(new_df[["num_dends", "num_soma", "trainable_params",
                  "test_acc", "test_loss"]].to_string(index=False))

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    if args.output is None:
        sys.exit("--output is required unless --dry-run is given")

    merged = dict(results)
    merged["testing"] = pd.concat(
        [reference_df, new_df[reference_df.columns]],
        ignore_index=True,
    )
    with open(args.output, "wb") as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nwrote {args.output}")
    print(f"models now present: {sorted(merged['testing']['model'].unique())}")


if __name__ == "__main__":
    main()