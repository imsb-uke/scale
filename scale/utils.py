import numpy as np
from scipy.spatial import KDTree
import torch
import scanpy as sc

from itertools import product

from collections import Counter

import random
import scipy

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from scipy.stats import entropy
from sklearn.metrics import silhouette_samples
from scipy.optimize import curve_fit


def preprocess(adata):
    if adata.X.max() < 10:
        print("The data seems to be already normalized")
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)


def centers2edgeindex(centers, thershold):
    """
    This function coverts centers to an edge index matrix (as a pytorch tensor)
    given a threshold.
    """

    # Craete a KDTree object
    kdt = KDTree(data=centers)

    # Find the edges with distances smaller than 'thershold' and build undirected'edge_index'
    edge_index = kdt.query_pairs(thershold)
    edge_index = list(edge_index)
    edge_index = torch.tensor(edge_index)
    edge_index = edge_index.T
    edge_index = torch.cat([edge_index, edge_index[(1, 0), :]], dim=1)

    # Calculate the pairwise distance and obtain the 'edge_weight' for 'edge_index'
    distance_mat = kdt.sparse_distance_matrix(kdt, thershold)
    edge_weight = [distance_mat[row[0], row[1]] for row in edge_index.T]
    edge_weight = torch.tensor(edge_weight)

    return edge_index, edge_weight


def spatial_graph(adata, method="knn", param=10, n_sample=None):
    """
    method: ['knn', 'distance']
    param:
        if method is knn, param is k
        if method is distance, param is distance
    """

    # Convert X and centers as torch.tensor
    if isinstance(adata.X, np.ndarray):
        X = adata.X
    elif isinstance(adata.X, scipy.sparse.csr_matrix):
        X = adata.X.toarray()
    else:
        raise ValueError(f"Unsupported data type: {type(adata.X)}")
    X = torch.tensor(X, dtype=torch.float32)
    X = X / (X.max(dim=0)[0] + 0.0000001)
    centers = adata.obsm["spatial"]
    centers = torch.tensor(centers, dtype=torch.float32)

    if method == "distance":
        dist = param
        edge_index, edge_weight = centers2edgeindex(centers, dist)
        # print(edge_index.shape)
        if n_sample is not None:
            n_nei = edge_index.shape[1]
            n_sample = torch.min(torch.tensor([n_sample, n_nei])).item()
            random_indices = torch.randperm(n_nei)[:n_sample]
            edge_index = edge_index[:, random_indices]
            edge_weight = edge_weight[random_indices]

        data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight)

    elif method == "knn":
        k = param
        knn_graph(adata, k)
        data = Data(
            x=X,
            edge_index=torch.tensor(adata.uns["graph"]["edge_index"]),
            edge_attr=torch.tensor(adata.uns["graph"]["edge_weight"]),
        )
    else:
        print("Method not implemented")

    return data


# Apply moving average on ARI
def smooth(x, N=10):
    x_smoothed = x.copy()
    for i in range(N):
        for j in range(len(x) - 1):
            x_smoothed[j] = x_smoothed[j + 1] = (x_smoothed[j] + x_smoothed[j + 1]) / 2
    return x_smoothed


# Sample from clusters by size
def cluster_sampler(cluster_labels, smallest_cluster_size: int = None):
    cluster_counts = Counter(cluster_labels)

    if smallest_cluster_size:
        print(smallest_cluster_size)
    else:
        smallest_cluster_size = min(cluster_counts.values())

    boolean_mask = [False] * len(cluster_labels)

    # A dictionary to keep track of sampled indices for each cluster
    sampled_indices = {cluster: [] for cluster in cluster_counts}

    # Sample clusters by the smallest cluster size
    for idx, label in enumerate(cluster_labels):
        if len(sampled_indices[label]) < smallest_cluster_size:
            sampled_indices[label].append(idx)

    # Update the boolean mask based on sampled indices
    for indices in sampled_indices.values():
        for idx in indices:
            boolean_mask[idx] = True

    return boolean_mask


def print_graph_stats(adata=None, edge_index=None, num_nodes=None, verbose=True):
    if adata is not None:
        edge_index = torch.from_numpy(adata.uns["graph"]["edge_index"])
        num_nodes = adata.shape[0]
    else:
        assert edge_index is not None, "Either adata or edge_index must be provided."

    if not verbose:
        return None

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    num_edges = edge_index.shape[1]
    in_degrees = pyg.utils.degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float)
    out_degrees = pyg.utils.degree(
        edge_index[0], num_nodes=num_nodes, dtype=torch.float
    )
    # check for self loops
    # n_self_loops = (edge_index[0] == edge_index[1]).sum().item()

    print("----------- Graph Stats -----------")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average in-degree: {in_degrees.mean().item()}")
    print(f"Average out-degree: {out_degrees.mean().item()}")
    print(f"Contains self-loops: {pyg.utils.contains_self_loops(edge_index)}")
    # print(f"Is undirected: {pyg.utils.is_undirected(edge_index)}")


def toarray(X):
    if isinstance(X, torch.Tensor):
        return X.cpu().numpy()
    else:
        try:
            return X.toarray()
        except Exception:
            return X


def save_graph_to_adata(adata, edge_index, edge_weight):
    adata.uns["graph"] = {}
    adata.uns["graph"]["edge_index"] = toarray(edge_index)
    adata.uns["graph"]["edge_weight"] = toarray(edge_weight)


def knn_graph(
    adata,
    knn,
    obsm_key="spatial",
    undirected=True,
    remove_self_loops=False,
    p=2,
    verbose=True,
):
    if obsm_key is not None:
        coords = adata.obsm[obsm_key]
    else:
        coords = toarray(adata.X)

    kdtree = scipy.spatial.KDTree(coords)
    distances, indices = kdtree.query(coords, k=knn + 1, p=p)
    edge_index = torch.cat(
        [
            torch.tensor(indices.flatten())[None, :],  # source
            torch.arange(0, coords.shape[0]).repeat_interleave(knn + 1)[
                None, :
            ],  # target
        ],
        axis=0,
    )
    edge_weight = torch.tensor(distances.flatten()).unsqueeze(-1)

    if undirected:
        edge_index, edge_weight = pyg.utils.to_undirected(edge_index, edge_weight)

    if remove_self_loops:
        edge_index, edge_weight = pyg.utils.remove_self_loops(edge_index, edge_weight)

    save_graph_to_adata(adata, edge_index, edge_weight)

    print_graph_stats(adata=adata, verbose=verbose)


def mixing_metrics_ebm(cluster_labels, batch_labels):
    # Entropy of Batch Mixing (EBM)
    entropies = []
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_batch_labels = batch_labels[cluster_indices]

        batch_counts = np.bincount(cluster_batch_labels)
        proportions = batch_counts / len(cluster_batch_labels)

        cluster_entropy = entropy(proportions, base=2)
        entropies.append(cluster_entropy)

    mean_entropy = np.mean(entropies)
    return mean_entropy


def mixing_metrics_asw(cluster_labels, batch_labels, data):
    # Average Silhouette Width (ASW)
    silhouette_values = silhouette_samples(data, batch_labels)
    mean_silhouette = np.mean(silhouette_values)
    return mean_silhouette


class GraphAggregation(MessagePassing):
    def __init__(self, aggr="mean"):
        super(GraphAggregation, self).__init__(aggr=aggr)

    def forward(self, x, edge_index, **kwargs):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


def curve_optimal_point(y):
    def D(x):
        x = np.array(x)
        return x[1 : len(x)] - x[0 : len(x) - 1]

    def sigmoid(x, a, b, c):
        return a / (1 + np.exp(-c * (x - b)))

    def curve_fitting(x, y):
        eps = 1e-6

        # Normalise time
        t_min = x.min()
        t_max = x.max()
        x = (x - t_min) / (t_max - t_min) + eps
        x = list(x)
        y = list(y)

        # Perform curve fitting
        initial_guess = (1, 1, 1)
        bounds = ([eps, eps, eps], [100, 100, 100])  # Lower bounds  # Upper bounds

        try:
            popt, pcov = curve_fit(sigmoid, (x), y, p0=initial_guess, bounds=bounds)
            a, b, c = tuple(popt)
        except Exception:
            a, b, c = (1, 1, 1, 1, 1)
        y_hat = sigmoid(x, a, b, c)

        return y_hat

    x = np.arange(len(y))
    y_hat = curve_fitting(x, y)

    max_ind = np.argmax(y)
    max_val = np.max(y)

    sat_ind = np.argmin(D(D(y_hat))) + 3
    sat_ind = sat_ind if sat_ind < len(y) else len(y) - 1
    sat_val = y[sat_ind]

    if (max_val - sat_val) > (max_val * 0.05):
        return max_ind
    else:
        return sat_ind


def select_best_lambdas(adata, del_other_embeddings=True):
    MI = adata.uns["scale"]["mi"]
    best_lambdas = [curve_optimal_point(MI[i]) for i in range(len(MI))]
    lambdas = np.array(adata.uns["scale"]["lambdas"])
    best_lambdas = lambdas[best_lambdas]
    if adata.uns["scale"].get("distances", None) is not None:
        params = list(adata.uns["scale"]["distances"])
        sparam_str = "dist"
    elif adata.uns["scale"].get("knn_values", None) is not None:
        params = list(adata.uns["scale"]["knn_values"])
        sparam_str = "knn"
    else:
        raise ValueError("No distances or knn values found in adata.uns['scale']")

    if del_other_embeddings:
        keys_to_del = []
        for param, lam in product(params, lambdas):
            # get index of param in params
            idx = params.index(param)
            if lam == best_lambdas[idx]:
                continue
            else:
                emb_key = to_emb_key(sparam_str, param, lam)
                keys_to_del.append(emb_key)
        adata.obsm = {k: v for k, v in adata.obsm.items() if k not in keys_to_del}
    adata.uns["scale"]["best_lambdas"] = best_lambdas


def to_emb_key(sparam_str, param, lam):
    return f"X_gnn_{sparam_str}_{param}_lam_{to_int(lam)}"


def to_int(x):
    if x.is_integer():
        return int(x)
    else:
        return x


def emb_key_to_params(emb_key):
    splits = emb_key.split("_")
    return splits[2], float(splits[3]), float(splits[5])


def extract_dist(x):
    return to_int(float(x.split("dist_")[-1].split("_")[0]))


def extract_knn(x):
    return int(x.split("knn_")[-1].split("_")[0])


def is_ordered(lst, ascending=True):
    return lst == sorted(lst, reverse=not ascending)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
