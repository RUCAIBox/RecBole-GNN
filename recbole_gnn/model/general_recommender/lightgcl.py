# -*- coding: utf-8 -*-
# @Time   : 2023/04/12
# @Author : Wanli Yang
# @Email  : 2013774@mail.nankai.edu.cn

r"""
LightGCL
################################################
Reference:
    Xuheng Cai et al. "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation" in ICLR 2023.

Reference code:
    https://github.com/HKUDS/LightGCL
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F


class LightGCL(GeneralRecommender):
    r"""LightGCL is a GCN-based recommender model.

    LightGCL guides graph augmentation by singular value decomposition (SVD) to not only
    distill the useful information of user-item interactions but also inject the global
    collaborative context into the representation alignment of contrastive learning.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCL, self).__init__(config, dataset)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]

        # load parameters info
        self.embed_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]
        self.temp = config["temp"]
        self.lambda_1 = config["lambda1"]
        self.lambda_2 = config["lambda2"]
        self.q = config["q"]
        self.act = nn.LeakyReLU(0.5)
        self.reg_loss = EmbLoss()

        # get the normalized adjust matrix
        self.adj_norm = self.coo2tensor(self.create_adjust_matrix())

        # perform svd reconstruction
        svd_u, s, svd_v = torch.svd_lowrank(self.adj_norm, q=self.q)
        self.u_mul_s = svd_u @ (torch.diag(s))
        self.v_mul_s = svd_v @ (torch.diag(s))
        del s
        self.ut = svd_u.T
        self.vt = svd_v.T

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.embed_dim)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.embed_dim)))
        self.E_u_list = [None] * (self.n_layers + 1)
        self.E_i_list = [None] * (self.n_layers + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (self.n_layers + 1)
        self.Z_i_list = [None] * (self.n_layers + 1)
        self.G_u_list = [None] * (self.n_layers + 1)
        self.G_i_list = [None] * (self.n_layers + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        self.E_u = None
        self.E_i = None
        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def create_adjust_matrix(self):
        r"""Get the normalized interaction matrix of users and items.

        Returns:
            coo_matrix of the normalized interaction matrix.
        """
        ratings = np.ones_like(self._user, dtype=np.float32)
        matrix = sp.csr_matrix(
            (ratings, (self._user, self._item)),
            shape=(self.n_users, self.n_items),
        ).tocoo()
        rowD = np.squeeze(np.array(matrix.sum(1)), axis=1)
        colD = np.squeeze(np.array(matrix.sum(0)), axis=0)
        for i in range(len(matrix.data)):
            matrix.data[i] = matrix.data[i] / pow(rowD[matrix.row[i]] * colD[matrix.col[i]], 0.5)
        return matrix

    def coo2tensor(self, matrix: sp.coo_matrix):
        r"""Convert coo_matrix to tensor.

        Args:
            matrix (scipy.coo_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        indices = torch.from_numpy(
            np.vstack((matrix.row, matrix.col)).astype(np.int64))
        values = torch.from_numpy(matrix.data)
        shape = torch.Size(matrix.shape)
        x = torch.sparse.FloatTensor(indices, values, shape).coalesce().to(self.device)
        return x

    def sparse_dropout(self, matrix, dropout):
        if dropout == 0.0:
            return matrix
        indices = matrix.indices()
        values = F.dropout(matrix.values(), p=dropout)
        size = matrix.size()
        return torch.sparse.FloatTensor(indices, values, size)

    def forward(self):
        for layer in range(1, self.n_layers + 1):
            # GNN propagation
            self.Z_u_list[layer] = torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout),
                                              self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1),
                                              self.E_u_list[layer - 1])
            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        # aggregate across layer
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        return self.E_u, self.E_i

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        E_u_norm, E_i_norm = self.forward()
        bpr_loss = self.calc_bpr_loss(E_u_norm, E_i_norm, user_list, pos_item_list, neg_item_list)
        ssl_loss = self.calc_ssl_loss(E_u_norm, E_i_norm, user_list, pos_item_list)
        total_loss = bpr_loss + ssl_loss
        return total_loss

    def calc_bpr_loss(self, E_u_norm, E_i_norm, user_list, pos_item_list, neg_item_list):
        r"""Calculate the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            E_u_norm (torch.Tensor): Ego embedding of all users after forwarding.
            E_i_norm (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = E_u_norm[user_list]
        pi_e = E_i_norm[pos_item_list]
        ni_e = E_i_norm[neg_item_list]
        pos_scores = torch.mul(u_e, pi_e).sum(dim=1)
        neg_scores = torch.mul(u_e, ni_e).sum(dim=1)
        loss1 = -(pos_scores - neg_scores).sigmoid().log().mean()

        # reg loss
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg *= self.lambda_2
        return loss1 + loss_reg

    def calc_ssl_loss(self, E_u_norm, E_i_norm, user_list, pos_item_list):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            E_u_norm (torch.Tensor): Ego embedding of all users in the original graph after forwarding.
            E_i_norm (torch.Tensor): Ego embedding of all items in the original graph after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """
        # calculate G_u_norm&G_i_norm
        for layer in range(1, self.n_layers + 1):
            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

        # aggregate across layer
        G_u_norm = sum(self.G_u_list)
        G_i_norm = sum(self.G_i_list)

        neg_score = torch.log(torch.exp(G_u_norm[user_list] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[pos_item_list] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[user_list] * E_u_norm[user_list]).sum(1) / self.temp, -5.0, 5.0)).mean() + (
            torch.clamp((G_i_norm[pos_item_list] * E_i_norm[pos_item_list]).sum(1) / self.temp, -5.0, 5.0)).mean()
        ssl_loss = -pos_score + neg_score
        return self.lambda_1 * ssl_loss

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
