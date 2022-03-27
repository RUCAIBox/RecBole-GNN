# @Time   : 2022/3/11
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
LESSR
################################################

Reference:
    Tianwen Chen and Raymond Chi-Wing Wong. "Handling Information Loss of Graph Neural Networks for Session-based Recommendation." in KDD 2020.

Reference code:
    https://github.com/twchen/lessr

"""

import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool
from recbole.model.abstract_recommender import SequentialRecommender


class EOPA(nn.Module):
    def __init__(
        self, input_dim, output_dim, batch_norm=True, feat_drop=0.0, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def reducer(self, nodes):
        m = nodes.mailbox['m']  # (num_nodes, deg, d)
        # m[i]: the messages passed to the i-th node with in-degree equal to 'deg'
        # the order of messages follows the order of incoming edges
        # since the edges are sorted by occurrence time when the EOP multigraph is built
        # the messages are in the order required by EOPA
        _, hn = self.gru(m)  # hn: (1, num_nodes, d)
        return {'neigh': hn.squeeze(0)}

    def forward(self, mg, feat):
        import dgl.function as fn

        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            mg.ndata['ft'] = self.feat_drop(feat)
            if mg.number_of_edges() > 0:
                mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
                neigh = mg.ndata['neigh']
                rst = self.fc_self(feat) + self.fc_neigh(neigh)
            else:
                rst = self.fc_self(feat)
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class SGAT(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, sg, feat):
        import dgl.ops as F

        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        q = self.fc_q(feat)
        k = self.fc_k(feat)
        v = self.fc_v(feat)
        e = F.u_add_v(sg, q, k)
        e = self.fc_e(torch.sigmoid(e))
        a = F.edge_softmax(sg, e)
        rst = F.u_mul_e_sum(sg, v, a)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.activation = activation

    def forward(self, g, feat, last_nodes, batch):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = torch.index_select(feat_v, dim=0, index=batch)
        e = self.fc_e(torch.sigmoid(feat_u + feat_v))
        alpha = softmax(e, batch)
        feat_norm = feat * alpha
        rst = global_add_pool(feat_norm, batch)
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class LESSR(SequentialRecommender):
    r"""LESSR analyzes the information losses when constructing session graphs,
    and emphasises lossy session encoding problem and the ineffective long-range dependency capturing problem.
    To solve the first problem, authors propose a lossless encoding scheme and an edge-order preserving aggregation layer.
    To solve the second problem, authors propose a shortcut graph attention layer that effectively captures long-range dependencies.

    Note:
        We follow the original implementation, which requires DGL package.
        We find it difficult to implement these functions via PyG, so we remain them.
        If you would like to test this model, please install DGL.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        embedding_dim = config['embedding_size']
        self.num_layers = config['n_layers']
        batch_norm = config['batch_norm']
        feat_drop = config['feat_drop']
        self.loss_type = config['loss_type']

        self.item_embedding = nn.Embedding(self.n_items, embedding_dim, max_norm=1)
        self.layers = nn.ModuleList()
        input_dim = embedding_dim
        for i in range(self.num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(embedding_dim),
                )
            else:
                layer = SGAT(
                    input_dim,
                    embedding_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(embedding_dim),
                )
            input_dim += embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.PReLU(embedding_dim),
        )
        input_dim += embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

    def forward(self, x, edge_index_EOP, edge_index_shortcut, batch, is_last):
        import dgl

        mg = dgl.graph((edge_index_EOP[0], edge_index_EOP[1]), num_nodes=batch.shape[0])
        sg = dgl.graph((edge_index_shortcut[0], edge_index_shortcut[1]), num_nodes=batch.shape[0])

        feat = self.item_embedding(x)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(mg, feat)
            else:
                out = layer(sg, feat)
            feat = torch.cat([out, feat], dim=1)
        sr_g = self.readout(mg, feat, is_last, batch)
        sr_l = feat[is_last]
        sr = torch.cat([sr_l, sr_g], dim=1)
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        sr = self.fc_sr(self.feat_drop(sr))
        return sr

    def calculate_loss(self, interaction):
        x = interaction['x']
        edge_index_EOP = interaction['edge_index_EOP']
        edge_index_shortcut = interaction['edge_index_shortcut']
        batch = interaction['batch']
        is_last = interaction['is_last']
        seq_output = self.forward(x, edge_index_EOP, edge_index_shortcut, batch, is_last)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        x = interaction['x']
        edge_index_EOP = interaction['edge_index_EOP']
        edge_index_shortcut = interaction['edge_index_shortcut']
        batch = interaction['batch']
        is_last = interaction['is_last']
        seq_output = self.forward(x, edge_index_EOP, edge_index_shortcut, batch, is_last)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index_EOP = interaction['edge_index_EOP']
        edge_index_shortcut = interaction['edge_index_shortcut']
        batch = interaction['batch']
        is_last = interaction['is_last']
        seq_output = self.forward(x, edge_index_EOP, edge_index_shortcut, batch, is_last)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
