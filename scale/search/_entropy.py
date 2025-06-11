import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from itertools import combinations
import scipy

from scale.utils import extract_dist, is_ordered


def find_best_settings(adata: AnnData, cutoff=None, top_n=None, min_dist=None):
    stability_df = adata.uns["scale"]["stability"]

    if min_dist is not None:
        stability_df.index = stability_df.index.astype(int)
        stability_df = stability_df.loc[stability_df.index >= min_dist, :]

    if cutoff is not None:
        mask = stability_df >= cutoff
        sub_df = mask.stack()[mask.stack()].reset_index()
        sub_df["setting"] = (
            "leiden_rep_0_dist_"
            + sub_df["distance"].astype(str)
            + "_res_"
            + sub_df["resolution"].astype(str)
        )
        settings = sub_df["setting"].tolist()
    elif top_n is not None:
        n_entries = stability_df.size

        # treat as fraction if top_n is less than 1
        if top_n < 1:
            top_n = int(top_n * n_entries)

        settings = (
            stability_df.stack()
            .nlargest(top_n)
            .index.map(lambda x: f"leiden_rep_0_dist_{x[0]}_res_{x[1]}")
            .tolist()
        )
    else:
        settings = (
            stability_df.stack()
            .index.map(lambda x: f"leiden_rep_0_dist_{x[0]}_res_{x[1]}")
            .tolist()
        )
    return settings


def filter_tuples(
    tuples,
    n_clusters_per_setting,
    enforce_dist_change=False,
    min_ncluster_increase_ratio=1.0,
    min_ncluster_increase=1,
    min_nclusters=1,
    max_nclusters=None,
    min_nclusters_start=None,
    max_nclusters_start=None,
    verbose=False,
    order=True,
):
    filtered_tuples = []
    for t in tuples:
        # order from low to high level or from high cluster num to low cluster num
        n_clusters = n_clusters_per_setting.loc[list(t)].sort_values(ascending=False)
        t_ordered = n_clusters.index.tolist()

        if n_clusters.duplicated().any():
            if verbose:
                print(f"Skipping tuple {n_clusters}\n")
                print("The number of clusters is duplicated")
            continue

        # check if the number of clusters increases enough
        min_ratio = (n_clusters / n_clusters.shift(-1)).min()
        if min_ratio < min_ncluster_increase_ratio:
            if verbose:
                print(f"Skipping tuple {n_clusters}\n")
                print(f"The number of clusters is not increasing enough: {min_ratio}")
            continue

        min_diff = (n_clusters - n_clusters.shift(-1)).min()
        if min_diff < min_ncluster_increase:
            if verbose:
                print(f"Skipping tuple {n_clusters}\n")
                print(f"The number of clusters is not increasing enough: {min_diff}")
            continue

        if min_nclusters is not None:
            if n_clusters.min() < min_nclusters:
                if verbose:
                    print(
                        f"Skipping tuple {n_clusters}\n"
                        f"The number of clusters is less than {min_nclusters}"
                    )
                continue
        if max_nclusters is not None:
            if n_clusters.max() > max_nclusters:
                if verbose:
                    print(
                        f"Skipping tuple {n_clusters}\n"
                        f"The number of clusters is greater than {max_nclusters}"
                    )
                continue

        if min_nclusters_start is not None:
            if n_clusters.iloc[-1] < min_nclusters_start:
                if verbose:
                    print(
                        f"Skipping tuple {n_clusters}\n"
                        f"The number of clusters is less than {min_nclusters_start}"
                    )
                continue
        if max_nclusters_start is not None:
            if n_clusters.iloc[-1] > max_nclusters_start:
                if verbose:
                    print(
                        f"Skipping tuple {n_clusters}\n"
                        f"The number of clusters is greater than {max_nclusters_start}"
                    )
                continue

        if enforce_dist_change:
            distances = [extract_dist(x) for x in n_clusters.index]
            if len(distances) != len(set(distances)):
                if verbose:
                    print(f"Skipping tuple {n_clusters}\n")
                    print("The distances are not unique")
                continue
            if not is_ordered(distances):
                if verbose:
                    print(f"Skipping tuple {n_clusters}\n")
                    print("The distances are not ordered")
                continue

        if order:
            filtered_tuples.append(t_ordered)
        else:
            filtered_tuples.append(t)

    return filtered_tuples


def calc_entropy_per_tuple(df):
    # iteratively select all pairs starting from the top
    t = df.columns.tolist()

    # t is ordered from high to low cluster number
    n_clusters = df.nunique()
    assert is_ordered(n_clusters.tolist(), ascending=False), (
        f"t is not ordered from high to low cluster number: {n_clusters}"
    )

    avg_entropies = []
    # iterate over all consecutive two-tuples
    for i in range(len(t) - 1):
        # select a two tuple
        sub_idx = t[i : i + 2]
        sub_df = df[sub_idx]

        # calc prob of an element in a lower level cluster orignating from a higher level cluster
        probs = (
            sub_df.groupby(sub_idx[0], observed=True)[sub_idx[1]]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .values
        )
        # probs.shape = (n_clusters_lower, n_clusters_higher)

        # calculate the entropy per row and sum up
        avg_entropy = scipy.stats.entropy(probs, axis=1).mean()
        avg_entropies.append(avg_entropy)
    agg_entropy = np.max(avg_entropies)

    return to_output(t, agg_entropy)


def calc_entropy(
    adata: AnnData,
    n_levels=2,
    cutoff=None,
    top_n=None,
    min_dist=None,
    min_cluster_size=None,
    n_jobs=1,
    **kwargs,
):
    settings = find_best_settings(adata, top_n=top_n, cutoff=cutoff, min_dist=min_dist)

    df = adata.obsm["scale_clusterings"]

    if min_cluster_size is not None:
        n_clusters_per_setting = []
        for setting in settings:
            value_counts = df[setting].value_counts()
            count_mask = value_counts >= min_cluster_size
            n_clusters = value_counts[count_mask].shape[0]
            n_clusters_per_setting.append(n_clusters)
        n_clusters_per_setting = pd.Series(n_clusters_per_setting, index=settings)
    else:
        n_clusters_per_setting = df[settings].nunique()

    tuples = list(combinations(settings, n_levels))
    tot_comps = len(tuples)
    print(f"Considering {len(settings)} settings and {tot_comps} tuples in total.")

    # filter tuples
    filtered_tuples = filter_tuples(
        tuples,
        n_clusters_per_setting,
        verbose=False,
        order=True,
        **kwargs,
    )

    print(f"Retained {len(filtered_tuples)} tuples after filtering")

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(calc_entropy_per_tuple)(df[t])
            for t in tqdm(filtered_tuples, total=len(filtered_tuples))
        )
    results = (
        pd.concat(results, axis=0).sort_values("avg_entropy").reset_index(drop=True)
    )
    # order from high to low level
    results = results[results.columns[::-1]]
    # if there are any tuples with entropy 0, select all of them
    mask = results["avg_entropy"] == 0
    if mask.sum() > 0:
        top_results = results[mask]
    else:
        row = results.loc[results["avg_entropy"].idxmin()]
        top_results = row.to_frame().T

    if len(top_results) > 1:
        print("Multiple top results found. Only storing the first one.")

    # store top results in anndata obs
    settings_cols = [c for c in top_results.columns if c.startswith("setting_")]
    for i, row in top_results.iterrows():
        # extract settings
        settings = row[settings_cols].tolist()
        # store clusterings in adata.obs
        col_names = [f"scale_l{i}_{setting}" for i, setting in enumerate(settings)]
        adata.obs[col_names] = adata.obsm["scale_clusterings"][settings]
        break

    return top_results


def to_output(t, avg_entropy):
    output = [[*t, avg_entropy]]
    output = pd.DataFrame(
        data=output,
        columns=[f"setting_{i}" for i in range(len(t))] + ["avg_entropy"],
        index=[0],
    )
    return output
