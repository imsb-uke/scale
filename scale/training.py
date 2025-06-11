import torch
import scanpy as sc
import numpy as np
from tqdm.auto import tqdm

from torch_geometric.utils import negative_sampling

from scale.model import GNN
from scale.utils import preprocess, spatial_graph, GraphAggregation, seed_everything


def train(adata, cfg, spatial_key="spatial", seed=200, device=None, return_model=False):
    seed_everything(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if cfg.preprocess:
        preprocess(adata)

    # calculate spatial graph for morans I and gearys c calculations
    sc.pp.neighbors(
        adata, use_rep=spatial_key, key_added=spatial_key, knn=True, n_neighbors=4
    )

    # setup training parameters
    distances = np.arange(
        cfg.distance_set.start, cfg.distance_set.stop, cfg.distance_set.step
    )
    knn_values = np.arange(cfg.knn_set.start, cfg.knn_set.stop, cfg.knn_set.step)
    lambda_set = cfg.lambda_set

    if cfg.spatial_graph_method == "distance":
        spatial_param_set = distances
        sparam_str = "dist"
    elif cfg.spatial_graph_method == "knn":
        spatial_param_set = knn_values
        sparam_str = "knn"
    else:
        raise ValueError(f"Invalid spatial graph method: {cfg.spatial_graph_method}")

    n_features = adata.X.shape[1]

    loss_1 = np.zeros((len(spatial_param_set), len(lambda_set)))
    loss_2 = np.zeros((len(spatial_param_set), len(lambda_set)))
    MI = np.zeros((len(spatial_param_set), len(lambda_set)))
    GC = np.zeros((len(spatial_param_set), len(lambda_set)))

    # training loop

    for i, param in tqdm(enumerate(spatial_param_set), total=len(spatial_param_set)):
        # Make a the pyg graph data
        data = spatial_graph(
            adata, method=cfg.spatial_graph_method, param=param, n_sample=cfg.n_sample
        ).to(device)
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.x.shape[0])

        # whether to aggregate the y values as target
        if cfg.y_aggregated:
            aggr_fn = GraphAggregation()
            Y_agg = aggr_fn(data.x, data.edge_index)
        else:
            Y_agg = None

        for j, lam in tqdm(enumerate(lambda_set), total=len(lambda_set)):
            model = GNN(
                n_input=n_features,
                n_hidden=cfg.n_hidden,
                n_heads=cfg.n_heads,
                n_batch=0,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

            ## Train the model
            for epoch in range(1, cfg.max_epoch + 1):
                if cfg.repeated_negative_sampling and epoch % 50 == 0:
                    neg_edge_index = negative_sampling(
                        pos_edge_index, num_nodes=data.x.shape[0]
                    )
                model, loss1, loss2 = train_step(
                    data,
                    pos_edge_index,
                    neg_edge_index,
                    model,
                    optimizer,
                    lam=lam,
                    print_loss=None,
                    epoch=100,
                    batch=None,
                    y_out=Y_agg,
                )

            model.eval()
            z, _, _ = model(data)
            z_np = z.to("cpu").detach().numpy()

            # store the embeddings in the adata object
            adata.obsm[f"X_gnn_{sparam_str}_{param}_lam_{lam}"] = z_np.copy()

            # track loss and spatial metrics
            loss_1[i, j] = loss1
            loss_2[i, j] = loss2
            MI[i, j] = sc.metrics.morans_i(
                adata.obsp["spatial_connectivities"],
                adata.obsm[f"X_gnn_{sparam_str}_{param}_lam_{lam}"].T,
            ).mean()
            GC[i, j] = sc.metrics.gearys_c(
                adata.obsp["spatial_connectivities"],
                adata.obsm[f"X_gnn_{sparam_str}_{param}_lam_{lam}"].T,
            ).mean()

        adata.uns["scale"] = {
            "loss1": loss_1,
            "loss2": loss_2,
            "mi": MI,
            "gc": GC,
            "lambdas": lambda_set,
        }
        if cfg.spatial_graph_method == "distance":
            adata.uns["scale"]["distances"] = distances
        elif cfg.spatial_graph_method == "knn":
            adata.uns["scale"]["knn_values"] = knn_values

        # store config
        adata.uns["scale"]["config"] = cfg.to_dict()

    if return_model:
        return model


def train_step(
    data,
    pos_edge_index,
    neg_edge_index,
    model,
    optimizer,
    lam,
    print_loss,
    epoch,
    batch=None,
    y_out=None,
):
    model.train()
    optimizer.zero_grad()
    z, _, y = model(data, batch=batch)
    loss1 = model.loss1(z, pos_edge_index, neg_edge_index)
    if y_out is None:
        loss2 = model.loss2(y, data.x)
    else:
        loss2 = model.loss2(y, y_out)
    loss = model.loss_total(loss1, loss2, lam)
    loss.backward()
    optimizer.step()
    if print_loss:
        if epoch % print_loss == 0:
            print(loss)
    return model, loss1, loss2
