import torch
from torch import nn
import torch_geometric.nn as gnn
from sklearn.metrics import average_precision_score, roc_auc_score


class GNNEncoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_heads, n_batch=0):
        super().__init__()
        self.n_batch = n_batch  # n_batch= -1: no batch norm, 0:  batch norm, >0: apply conditional batch norm
        self.conv = gnn.GATv2Conv(
            in_channels=n_input, out_channels=n_hidden, heads=n_heads
        )
        self.linear = nn.Linear(
            in_features=n_heads * n_hidden, out_features=n_heads * n_hidden
        )
        self.relu = nn.ReLU()

        if n_batch > -1:
            self.bn = nn.BatchNorm1d(n_heads * n_hidden, affine=True)

        if n_batch > 0:
            self.gamma_layer = nn.Linear(n_batch, n_heads * n_hidden)
            self.beta_layer = nn.Linear(n_batch, n_heads * n_hidden)

    def forward(self, data, batch=None):
        X, edge_index, _ = (
            data.x,
            data.edge_index,
            data.edge_attr.unsqueeze(-1).float(),
        )

        H = self.conv(X, edge_index, return_attention_weights=None)
        H = self.relu(H)

        if self.n_batch > -1:
            if batch is not None:
                gamma = self.gamma_layer(batch.unsqueeze(1).float())
                beta = self.beta_layer(batch.unsqueeze(1).float())
                H = self.bn(H)
                H = gamma * H + beta
            else:
                H = self.bn(H)

        H = self.linear(H)
        return H


class EdgeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([1.0]))
        self.relu = nn.ReLU()

    def forward(self, z, edge_index, add_linear=True, activation="sigmoid"):
        dist = ((z[edge_index[0]] - z[edge_index[1]]) ** 2).sum(dim=1)

        if add_linear:
            dist = self.relu(self.a) * dist + self.b

        if activation == "linear":
            out = -dist
        elif activation == "exp":
            out = torch.exp(-dist)
        elif activation == "sigmoid":
            out = torch.sigmoid(-dist)
        else:
            print("activation not defined")

        return out


class LinearDecoder(nn.Module):
    def __init__(self, n_hidden, n_output):
        super().__init__()
        self.layer = nn.Linear(in_features=n_hidden, out_features=n_output)

    def forward(self, z):
        value = self.layer(z)
        return value


class GNN(nn.Module):
    def __init__(self, n_input, n_hidden=10, n_heads=5, n_batch=0):
        super().__init__()
        self.encoder = GNNEncoder(n_input, n_hidden, n_heads, n_batch)
        self.decoder1 = EdgeDecoder()
        self.decoder2 = LinearDecoder(n_hidden=n_hidden * n_heads, n_output=n_input)
        self.mse_loss = nn.MSELoss()

    def forward(self, data, batch=None):
        z = self.encoder(data, batch=batch)
        value1 = self.decoder1(
            z, data.edge_index, add_linear=True, activation="sigmoid"
        )
        value2 = self.decoder2(z)
        return z, value1, value2

    def encode(self, data, batch=None, return_attention_weights=False):
        z = self.encoder(
            data, batch=batch, return_attention_weights=return_attention_weights
        )
        return z

    def loss1(self, z, pos_edge_index, neg_edge_index):
        pos_loss = -torch.log(
            self.decoder1(z, pos_edge_index, add_linear=True, activation="sigmoid")
            + 0.0000001
        ).mean()
        neg_loss = -torch.log(
            1
            - self.decoder1(z, neg_edge_index, add_linear=True, activation="sigmoid")
            + 0.0000001
        ).mean()
        return pos_loss + neg_loss

    def loss2(self, y_hat, y):
        loss = self.mse_loss(y_hat, y)
        return loss

    def loss_total(self, L1, L2, lam):
        return lam * L1 + L2

    def link_test(self, z, pos_edge_index, neg_edge_index, batch=None):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder1(
            z, pos_edge_index, add_linear=True, activation="sigmoid"
        )
        neg_pred = self.decoder1(
            z, neg_edge_index, add_linear=True, activation="sigmoid"
        )
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
