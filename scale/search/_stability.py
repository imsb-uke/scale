import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm
from anndata import AnnData
import re


def calc_stability(
    adata: AnnData,
    n_repeat: int | None = None,
    verbose: int | bool = False,
    min_dist: float | None = None,
    max_dist: float | None = None,
    min_res: float | None = None,
    max_res: float | None = None,
):
    assert "scale_clusterings" in adata.obsm, (
        "scale_clusterings not found in adata.obsm"
    )
    df = adata.obsm["scale_clusterings"]
    columns = df.columns

    # extract all possible resolutions
    resolutions = sorted(
        list(set([x.split("res_")[-1].split("_")[0] for x in columns]))
    )

    if min_res is not None:
        resolutions = [r for r in resolutions if float(r) >= min_res]

    if max_res is not None:
        resolutions = [r for r in resolutions if float(r) <= max_res]

    # extract all possible distances
    distances = sorted(list(set([x.split("dist_")[-1].split("_")[0] for x in columns])))

    if min_dist is not None:
        distances = [d for d in distances if float(d) >= min_dist]

    if max_dist is not None:
        distances = [d for d in distances if float(d) <= max_dist]

    # extract number of repetitions
    if n_repeat is None:
        # find the smallest number of repetitions present for all clusterings
        settings = [re.sub(r"rep_\d+_", "", x) for x in columns]
        tmp = pd.DataFrame({"settings": settings})
        n_repeat = tmp["settings"].value_counts().min()

    if verbose:
        print(f"n_repeat: {n_repeat}")
        print(f"n_resolutions: {len(resolutions)}")
        print(f"n_distances: {len(distances)}")
        print(f"resolutions: {resolutions}")
        print(f"distances: {distances}")

    stability_df = pd.DataFrame(
        np.zeros((len(distances), len(resolutions))),
        index=distances,
        columns=resolutions,
    )
    stability_df.index.name = "distance"
    stability_df.columns.name = "resolution"
    for i, dist in tqdm(
        enumerate(distances), total=len(distances), desc="Calculating stability"
    ):
        for j, res in enumerate(resolutions):
            ari_scores = []
            for r1 in range(n_repeat):
                for r2 in range(r1 + 1, n_repeat):
                    try:
                        ari = adjusted_rand_score(
                            df[f"leiden_rep_{r1}_dist_{dist}_res_{res}"],
                            df[f"leiden_rep_{r2}_dist_{dist}_res_{res}"],
                        )
                    except Exception as e:
                        print("Error:", e)
                        ari = 0
                    ari_scores.append(ari)
            ari_scores = np.array(ari_scores)
            stability_df.loc[dist, res] = ari_scores.mean()

    for col in stability_df.columns:
        stability_df[col] = stability_df[col].astype(float)

    adata.uns["scale"]["stability"] = stability_df
