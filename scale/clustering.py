import scanpy as sc
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
)

from scale.config import Config


def calc_clusterings(
    adata,
    n_jobs=20,
    spatial_key="spatial",
    **kwargs,
):
    cfg = Config(adata.uns["scale"]["config"])
    resolutions = np.arange(
        cfg.resolution_set.start, cfg.resolution_set.stop, cfg.resolution_set.step
    ).round(4)

    all_clusterings = pd.DataFrame(index=adata.obs_names)

    emb_keys = [k for k in adata.obsm.keys() if "X_gnn" in k]

    for emb_key in emb_keys:
        ad_tmp = sc.AnnData(adata.obsm[emb_key])
        ad_tmp.obs = pd.DataFrame(
            adata.obsm[spatial_key], columns=["x", "y"], index=adata.obs_names
        )
        sc.pp.neighbors(ad_tmp, use_rep="X")
        dist = emb_key.split("dist_")[-1].split("_lam")[0]
        for i in tqdm(range(cfg.n_repeats), desc="Calculating clusterings"):
            parallel_leiden(
                ad_tmp,
                resolutions,
                key_added=f"leiden_rep_{i}_dist_{dist}",
                n_jobs=n_jobs,
                verbose=kwargs.get("verbose", False),
                random_state=i,
                **kwargs,
            )
        clusterings = ad_tmp.obs[[c for c in ad_tmp.obs.columns if "leiden" in c]]
        all_clusterings = pd.concat([all_clusterings, clusterings], axis=1)
    adata.obsm["scale_clusterings"] = all_clusterings


def parallel_leiden(
    adata,
    resolutions,
    key_added="scale",
    n_jobs=10,
    verbose=True,
    random_state=0,
    **kwargs,
):
    """
    Perform Leiden clustering with different resolutions in parallel and
    add result as columns to adata.obs 'in_place'.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    resolutions : list
        List of resolution parameters for Leiden clustering.
    key_added : str, optional (default: "cluster")
        Key under which to add the cluster labels to adata.obs.
        Final keys will be {key_added}_res_{resolution}.
    n_jobs : int, optional (default: 10)
        Number of parallel jobs to run.
    verbose : bool, optional (default: True)
        Print progress messages.
    random_state : int, optional (default: 0)
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to scanpy.tl.leiden().

    Returns
    -------
    adata : AnnData
        The updated AnnData object with clustering results added to obs.
    """

    def loop(r, adata):
        if verbose:
            print(f"Resolution = {r} Started!")
        sc.tl.leiden(
            adata,
            resolution=r,
            key_added=key_added + "_res_" + str(r),
            random_state=random_state,
            **kwargs,
        )
        if verbose:
            print(f"Resolution = {r} Done!")
        return adata.obs[key_added + "_res_" + str(r)]

    clusterings = Parallel(n_jobs=n_jobs)(delayed(loop)(r, adata) for r in resolutions)

    for clustering in clusterings:
        adata.obs[clustering.name] = clustering

    return adata


def calc_cluster_metrics(
    labels_true,
    labels_pred,
    metrics=["nmi", "ami", "ari", "hom", "com", "fmi"],
    verbose=False,
):
    metrics_map = {
        "nmi": normalized_mutual_info_score,
        "ami": adjusted_mutual_info_score,
        "ari": adjusted_rand_score,
        "hom": homogeneity_score,
        "com": completeness_score,
        "fmi": fowlkes_mallows_score,
    }

    results = []
    for metric in metrics:
        func = metrics_map[metric]
        results.append(func(labels_true, labels_pred))
    if verbose:
        ", ".join(
            [f"{metric}: {result:.3f}" for metric, result in zip(metrics, results)]
        )
    return results
