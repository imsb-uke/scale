import numpy as np

from scale.utils import (
    select_best_lambdas,
    spatial_sample_split,
    preprocess,
    check_for_raw_counts,
)
from scale.training import train
from scale.clustering import calc_clusterings
from scale.search import calc_stability, calc_entropy
from scale.config import Config
import scanpy as sc
import pandas as pd
import scanpy.external as sce


def run_scale(
    adata,
    cfg: Config,
    sample_key: str | None = None,
    integration_method: str | None = None,
    layer: str | None = None,
    celltype_key: str | None = None,
    **kwargs,
):
    spatial_key = kwargs.get("spatial_key", "spatial")
    if sample_key is not None:
        spatial_sample_split(
            adata,
            sample_key,
            in_key=spatial_key,
            out_key=spatial_key,
            displacement=kwargs.get("displacement", 1000),
        )

        if integration_method == "harmony":
            # prepare object and call run_scale without the sample key
            adata.X = adata.layers[layer].copy() if layer else adata.X
            preprocess(adata)
            sc.pp.pca(adata)
            sce.pp.harmony_integrate(adata, sample_key)
            ad_tmp = sc.AnnData(
                X=adata.obsm["X_pca_harmony"],
                obs=adata.obs[[sample_key]],
                obsm={spatial_key: adata.obsm[spatial_key]},
            )
            print("✅ Successfully ran harmony integration!")
            cfg.preprocess = False
            run_scale(
                ad_tmp,
                cfg,
                sample_key=None,
                integration_method=None,
                layer=None,
                **kwargs,
            )
            # integrate the results back into the original object
            transfer_scale_results(ad_tmp, adata)
        elif integration_method == "celltype_expression":
            assert celltype_key is not None, (
                "celltype_key is required for cell type expression integration"
            )
            add_ct_expression_layer(
                adata, celltype_key, sample_key, layer_in=layer, layer_out="X_ct"
            )

            cfg.preprocess = True
            run_scale(
                adata,
                cfg,
                sample_key=None,
                integration_method=None,
                layer="X_ct",
                **kwargs,
            )
        elif integration_method == "celltype_onehot":
            assert celltype_key is not None, (
                "celltype_key is required for cell type one-hot integration"
            )
            raise NotImplementedError("celltype_onehot integration not implemented")
        else:
            raise ValueError(f"Invalid integration method: {integration_method}")
    else:
        train(adata, cfg, layer=layer, spatial_key=spatial_key)
        select_best_lambdas(adata)

        calc_clusterings(
            adata,
            flavor=kwargs.get("flavor", "igraph"),
            n_iterations=kwargs.get("n_iterations", 2),
        )

        calc_stability(
            adata,
            verbose=kwargs.get("verbose", True),
            n_repeat=kwargs.get("n_repeat", 4),
            min_dist=kwargs.get("min_dist", 15),
            max_dist=kwargs.get("max_dist", 55),
        )
        top_results = calc_entropy(
            adata,
            n_levels=kwargs.get("n_levels", 2),
            top_n=kwargs.get("top_n", 0.15),
            enforce_dist_change=kwargs.get("enforce_dist_change", False),
            min_ncluster_increase=kwargs.get("min_ncluster_increase", 20),
            min_nclusters_start=kwargs.get("min_nclusters_start", 3),
        )


def transfer_scale_results(from_adata, to_adata):
    scale_cols = [c for c in from_adata.obs.columns if c.startswith("scale_l")]
    to_adata.obs[scale_cols] = from_adata.obs[scale_cols]
    to_adata.uns["scale"] = from_adata.uns["scale"]
    for k, v in from_adata.obsm.items():
        if k == "spatial":
            continue
        to_adata.obsm[k] = v


def get_ct_expression(adata, obs_val, sample_val, obs_key, sample_key, layer_in=None):
    mask = (adata.obs[obs_key] == obs_val) & (adata.obs[sample_key] == sample_val)
    X = adata[mask, :].X if layer_in is None else adata[mask, :].layers[layer_in]
    check_for_raw_counts(X)
    X = X.mean(axis=0)
    return np.asarray(X).flatten()


def add_ct_expression_layer(
    adata, celltype_key, sample_key, layer_in=None, layer_out="X_ct"
):
    # if there is a sample with all cell types use that as the reference
    # if not select the one with the most cell types and add information from other samples for all missing ones
    ct_counts = (
        adata.obs.groupby(sample_key, observed=True)[celltype_key]
        .value_counts()
        .unstack()
        .fillna(0)
    )
    n_cts = ct_counts.astype(bool).astype(int).sum(axis=1)

    ref_sample = n_cts.idxmax()
    # add information from other samples for all missing ones
    vector_map = {}
    for col in ct_counts.columns:
        if ct_counts.loc[ref_sample, col] == 0:
            vector_map[col] = get_ct_expression(
                adata,
                col,
                ct_counts[col].idxmax(),
                celltype_key,
                sample_key,
                layer_in,
            )
        else:
            vector_map[col] = get_ct_expression(
                adata, col, ref_sample, celltype_key, sample_key, layer_in
            )
    X = np.array([vector_map[ct] for ct in adata.obs[celltype_key]])
    adata.layers[layer_out] = X


def add_ct_onehot_encoding(adata, celltype_key, obsm_out="ct_onehot"):
    df = pd.get_dummies(adata.obs[celltype_key], dtype=np.int8)
    df.index = adata.obs_names
    adata.obsm[obsm_out] = df.values


def create_ct_onehot_adata(adata, celltype_key):
    df = pd.get_dummies(adata.obs[celltype_key], dtype=np.int8)

    return sc.AnnData(
        X=df.values,
        var=pd.DataFrame(index=df.columns),
        obs=adata.obs,
        obsm=adata.obsm,
        uns=adata.uns,
    )


def prepare_multi_sample_integration(
    adata,
    sample_key,
    obs_key=None,
    kind="ct_expression",
    layer_in=None,
    layer_out="X_agg",
    spatial_key="spatial",
    displacement=1000,
):
    spatial_sample_split(
        adata,
        sample_key,
        in_key=spatial_key,
        out_key=spatial_key,
        displacement=displacement,
    )

    if kind == "ct_expression":
        # if there is a sample with all cell types use that as the reference
        # if not select the one with the most cell types and add information from other samples for all missing ones
        ct_counts = (
            adata.obs.groupby(sample_key, observed=True)[obs_key]
            .value_counts()
            .unstack()
            .fillna(0)
        )
        n_cts = ct_counts.astype(bool).astype(int).sum(axis=1)

        ref_sample = n_cts.idxmax()
        # add information from other samples for all missing ones
        vector_map = {}
        for col in ct_counts.columns:
            if ct_counts.loc[ref_sample, col] == 0:
                vector_map[col] = get_ct_expression(
                    adata, col, ct_counts[col].idxmax(), obs_key, sample_key, layer_in
                )
            else:
                vector_map[col] = get_ct_expression(
                    adata, col, ref_sample, obs_key, sample_key, layer_in
                )
        X = np.array([vector_map[ct] for ct in adata.obs[obs_key]])
        adata.layers[layer_out] = X
        return adata
    elif kind == "ct_onehot":
        df = pd.get_dummies(adata.obs[obs_key], dtype=np.int8)
        X = df.values
        var = pd.DataFrame(index=df.columns)

        ad_tmp = sc.AnnData(
            X=X,
            obs=adata.obs[[sample_key]],
            obsm={spatial_key: adata.obsm[spatial_key]},
            var=var,
        )

        # optionally run pca
        sc.pp.pca(ad_tmp, n_comps=50)

        ad_tmp = sc.AnnData(
            X=ad_tmp.obsm["X_pca"],
            obs=adata.obs[[sample_key]],
            obsm={spatial_key: adata.obsm[spatial_key]},
        )
        return ad_tmp
    elif kind == "harmony":
        preprocess(adata)
        sc.pp.pca(adata)
        sce.pp.harmony_integrate(adata, sample_key)
        ad_tmp = sc.AnnData(
            X=adata.obsm["X_pca_harmony"],
            obs=adata.obs[[sample_key]],
            obsm={spatial_key: adata.obsm[spatial_key]},
        )
        return ad_tmp
    else:
        raise ValueError(f"Invalid kind: {kind}")
