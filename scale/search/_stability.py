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
    min_knn: int | None = None,
    max_knn: int | None = None,
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

    # extract all possible 
    sparam_str = "dist" if "dist" in columns[0] else "knn"
    sparam_values = sorted(list(set([x.split(f"{sparam_str}_")[-1].split("_")[0] for x in columns])), key=float)

    
    def check_bounds(sparam_values, min_sparam, max_sparam):
        if min_sparam is not None:
            sparam_values = [d for d in sparam_values if float(d) >= min_sparam]
        if max_sparam is not None:
            sparam_values = [d for d in sparam_values if float(d) <= max_sparam]
        return sparam_values
    
    if sparam_str == "dist":
        sparam_values = check_bounds(sparam_values, min_dist, max_dist)
    elif sparam_str == "knn":
        sparam_values = check_bounds(sparam_values, min_knn, max_knn)
    else:
        raise ValueError(f"Invalid sparam_str: {sparam_str}")

    # extract number of repetitions
    if n_repeat is None:
        # find the smallest number of repetitions present for all clusterings
        settings = [re.sub(r"rep_\d+_", "", x) for x in columns]
        tmp = pd.DataFrame({"settings": settings})
        n_repeat = tmp["settings"].value_counts().min()

    if verbose:
        print(f"n_repeat: {n_repeat}")
        print(f"n_resolutions: {len(resolutions)}")
        print(f"n_{sparam_str}_values: {len(sparam_values)}")
        print(f"resolutions: {resolutions}")
        print(f"{sparam_str}_values: {sparam_values}")

    stability_df = pd.DataFrame(
        np.zeros((len(sparam_values), len(resolutions))),
        index=sparam_values,
        columns=resolutions,
    )
    stability_df.index.name = sparam_str
    stability_df.columns.name = "resolution"
    for i, sparam_value in tqdm(
        enumerate(sparam_values), total=len(sparam_values), desc="Calculating stability"
    ):
        for j, res in enumerate(resolutions):
            ari_scores = []
            for r1 in range(n_repeat):
                for r2 in range(r1 + 1, n_repeat):
                    try:
                        ari = adjusted_rand_score(
                            df[f"leiden_rep_{r1}_{sparam_str}_{sparam_value}_res_{res}"],
                            df[f"leiden_rep_{r2}_{sparam_str}_{sparam_value}_res_{res}"],
                        )
                    except Exception as e:
                        print("Error:", e)
                        ari = 0
                    ari_scores.append(ari)
            ari_scores = np.array(ari_scores)
            stability_df.loc[sparam_value, res] = ari_scores.mean()

    for col in stability_df.columns:
        stability_df[col] = stability_df[col].astype(float)

    adata.uns["scale"]["stability"] = stability_df
