# -*- coding: utf-8 -*-
# @Time   : 2022/3/8
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com
r"""
SGL
################################################
Reference:
    Jiancan Wu et al. "SGL: Self-supervised Graph Learning for Recommendation" in SIGIR 2021.

Reference code:
    https://github.com/wujcan/SGL
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class SGL(GeneralGraphRecommender):
    r"""SGL is a GCN-based recommender model.

    SGL supplements the classical supervised task of recommendation with an auxiliary
    self supervised task, which reinforces node representation learning via self-
    discrimination.Specifically,SGL generates multiple views of a node, maximizing the
    agreement between different views of the same node compared to that of other nodes.
    SGL devises three operators to generate the views — node dropout, edge dropout, and
    random walk — that change the graph structure in different manners.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config["embedding_size"]
        self.n_layers = int(config["n_layers"])
        self.aug_type = config["type"]
        self.drop_ratio = config["drop_ratio"]
        self.ssl_tau = config["ssl_tau"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]

        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        self.dataset = dataset

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def train(self, mode: bool = True):
        r"""Override train method of base class. The subgraph is reconstructed each time it is called.

        """
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T

    def graph_construction(self):
        r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node.

        """
        if self.aug_type == "ND" or self.aug_type == "ED":
            self.sub_graph1 = [self.random_graph_augment()] * self.n_layers
            self.sub_graph2 = [self.random_graph_augment()] * self.n_layers
        elif self.aug_type == "RW":
            self.sub_graph1 = [self.random_graph_augment() for _ in range(self.n_layers)]
            self.sub_graph2 = [self.random_graph_augment() for _ in range(self.n_layers)]

    def random_graph_augment(self):
        def rand_sample(high, size=None, replace=True):
            return np.random.choice(np.arange(high), size=size, replace=replace)

        if self.aug_type == "ND":
            drop_user = rand_sample(self.n_users, size=int(self.n_users * self.drop_ratio), replace=False)
            drop_item = rand_sample(self.n_items, size=int(self.n_items * self.drop_ratio), replace=False)

            mask = np.isin(self._user.numpy(), drop_user)
            mask |= np.isin(self._item.numpy(), drop_item)
            keep = np.where(~mask)

            row = self._user[keep]
            col = self._item[keep] + self.n_users

        elif self.aug_type == "ED" or self.aug_type == "RW":
            keep = rand_sample(len(self._user), size=int(len(self._user) * (1 - self.drop_ratio)), replace=False)
            row = self._user[keep]
            col = self._item[keep] + self.n_users

        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_weight = torch.ones(edge_index.size(1))
        num_nodes = self.n_users + self.n_items

        if self.use_sparse:
            adj_t = self.dataset.edge_index_to_adj_t(edge_index, edge_weight, num_nodes, num_nodes)
            adj_t = gcn_norm(adj_t, None, num_nodes, add_self_loops=False)
            return adj_t.to(self.device), None

        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=False)

        return edge_index.to(self.device), edge_weight.to(self.device)

    def forward(self, graph=None):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]

        if graph is None:  # for the original graph
            for _ in range(self.n_layers):
                all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
                embeddings_list.append(all_embeddings)
        else:  # for the augmented graph
            for graph_edge_index, graph_edge_weight in graph:
                all_embeddings = self.gcn_conv(all_embeddings, graph_edge_index, graph_edge_weight)
                embeddings_list.append(all_embeddings)

        embeddings_list = torch.stack(embeddings_list, dim=1)
        embeddings_list = torch.mean(embeddings_list, dim=1, keepdim=False)
        user_all_embeddings, item_all_embeddings = torch.split(embeddings_list, [self.n_users, self.n_items], dim=0)

        return user_all_embeddings, item_all_embeddings

    def calc_bpr_loss(self, user_emd, item_emd, user_list, pos_item_list, neg_item_list):
        r"""Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)

        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))

        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2, dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]

        user_emd, item_emd = self.forward()
        user_sub1, item_sub1 = self.forward(self.sub_graph1)
        user_sub2, item_sub2 = self.forward(self.sub_graph2)

        total_loss = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list) + \
            self.calc_ssl_loss(user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2)
        return total_loss

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)
