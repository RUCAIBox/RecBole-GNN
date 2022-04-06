# @Time   : 2022/4/5
# @Author : Lanling Xu
# @Email  : xulanling_sherry@163.com

r"""
MHCN
################################################
Reference:
    Junliang Yu et al. "Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation." in WWW 2021.

Reference code:
    https://github.com/Coder-Yu/QRec
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import coo_matrix

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import SocialRecommender
from recbole_gnn.model.layers import BipartiteGCNConv


class GatingLayer(nn.Module):
    def __init__(self, dim):
        super(GatingLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(self.dim, self.dim)
        self.activation = nn.Sigmoid()

    def forward(self, emb):
        embedding = self.linear(emb)
        embedding = self.activation(embedding)
        embedding = torch.mul(emb, embedding)
        return embedding


class AttLayer(nn.Module):
    def __init__(self, dim):
        super(AttLayer, self).__init__()
        self.dim = dim
        self.attention_mat = nn.Parameter(torch.randn([self.dim, self.dim]))
        self.attention = nn.Parameter(torch.randn([1, self.dim]))

    def forward(self, *embs):
        weights = []
        emb_list = []
        for embedding in embs:
            weights.append(torch.sum(torch.mul(self.attention, torch.matmul(embedding, self.attention_mat)), dim=1))
            emb_list.append(embedding)
        score = torch.nn.Softmax(dim=0)(torch.stack(weights, dim=0))
        embeddings = torch.stack(emb_list, dim=0)
        mixed_embeddings = torch.mul(embeddings, score.unsqueeze(dim=2).repeat(1, 1, self.dim)).sum(dim=0)
        return mixed_embeddings


class MHCN(SocialRecommender):
    r"""MHCN fuses hypergraph modeling and graph neural networks in social recommendation by 
    exploiting multiple types of high-order user relations under a multi-channel setting.
    
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MHCN, self).__init__(config, dataset)

        # load dataset info
        self.R_user_edge_index, self.R_user_edge_weight, self.R_item_edge_index, self.R_item_edge_weight = self.get_bipartite_inter_mat(dataset)
        H_s, H_j, H_p = self.get_motif_adj_matrix(dataset)

        # transform matrix to edge index and edge weight for convolution
        self.H_s_edge_index, self.H_s_edge_weight = self.get_edge_index_weight(H_s)
        self.H_j_edge_index, self.H_j_edge_weight = self.get_edge_index_weight(H_j)
        self.H_p_edge_index, self.H_p_edge_weight = self.get_edge_index_weight(H_p)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.ssl_reg = config['ssl_reg']
        self.reg_weight = config['reg_weight']

        # define embedding and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.bipartite_gcn_conv = BipartiteGCNConv(dim=self.embedding_size)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # define gating layers
        self.gating_c1 = GatingLayer(self.embedding_size)
        self.gating_c2 = GatingLayer(self.embedding_size)
        self.gating_c3 = GatingLayer(self.embedding_size)
        self.gating_simple = GatingLayer(self.embedding_size)

        # define self supervised gating layers
        self.ss_gating_c1 = GatingLayer(self.embedding_size)
        self.ss_gating_c2 = GatingLayer(self.embedding_size)
        self.ss_gating_c3 = GatingLayer(self.embedding_size)

        # define attention layers
        self.attention_layer = AttLayer(self.embedding_size)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_bipartite_inter_mat(self, dataset):
        R_user_edge_index, R_user_edge_weight = dataset.get_bipartite_inter_mat(row='user', row_norm=False)
        R_item_edge_index, R_item_edge_weight = dataset.get_bipartite_inter_mat(row='item', row_norm=False)
        return R_user_edge_index.to(self.device), R_user_edge_weight.to(self.device), R_item_edge_index.to(self.device), R_item_edge_weight.to(self.device)

    def get_edge_index_weight(self, matrix):
        matrix = coo_matrix(matrix)
        edge_index = torch.stack([torch.LongTensor(matrix.row), torch.LongTensor(matrix.col)])
        edge_weight = torch.FloatTensor(matrix.data)
        return edge_index.to(self.device), edge_weight.to(self.device)

    def get_motif_adj_matrix(self, dataset):
        S = dataset.net_matrix()
        Y = dataset.inter_matrix()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10  = Y.dot(Y.T) - A8 - A9
        # addition and row-normalization
        H_s = sum([A1, A2, A3, A4, A5, A6, A7])
        # add epsilon to avoid divide by zero Warning
        H_s = H_s.multiply(1.0 / (H_s.sum(axis=1) + 1e-7).reshape(-1, 1))
        H_j = sum([A8, A9])
        H_j = H_j.multiply(1.0 / (H_j.sum(axis=1) + 1e-7).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p > 1)
        H_p = H_p.multiply(1.0 / (H_p.sum(axis=1) + 1e-7).reshape(-1, 1))
        return H_s, H_j, H_p

    def forward(self):
        # get ego embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # self-gating
        user_embeddings_c1 = self.gating_c1(user_embeddings)
        user_embeddings_c2 = self.gating_c2(user_embeddings)
        user_embeddings_c3 = self.gating_c3(user_embeddings)
        simple_user_embeddings = self.gating_simple(user_embeddings)

        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        all_embeddings_i = [item_embeddings]

        for layer_idx in range(self.n_layers):
            mixed_embedding = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3) + simple_user_embeddings / 2
            
            # Channel S
            user_embeddings_c1 = self.bipartite_gcn_conv((user_embeddings_c1, user_embeddings_c1), self.H_s_edge_index.flip([0]), self.H_s_edge_weight, size=(self.n_users, self.n_users))
            norm_embeddings = F.normalize(user_embeddings_c1, p=2, dim=1)
            all_embeddings_c1 += [norm_embeddings]

            # Channel J
            user_embeddings_c2 = self.bipartite_gcn_conv((user_embeddings_c2, user_embeddings_c2), self.H_j_edge_index.flip([0]), self.H_j_edge_weight, size=(self.n_users, self.n_users))
            norm_embeddings = F.normalize(user_embeddings_c2, p=2, dim=1)
            all_embeddings_c2 += [norm_embeddings]

            # Channel P
            user_embeddings_c3 = self.bipartite_gcn_conv((user_embeddings_c3, user_embeddings_c3), self.H_p_edge_index.flip([0]), self.H_p_edge_weight, size=(self.n_users, self.n_users))
            norm_embeddings = F.normalize(user_embeddings_c3, p=2, dim=1)
            all_embeddings_c3 += [norm_embeddings]

            # item convolution
            new_item_embeddings = self.bipartite_gcn_conv((mixed_embedding, item_embeddings), self.R_item_edge_index.flip([0]), self.R_item_edge_weight, size=(self.n_users, self.n_items))
            norm_embeddings = F.normalize(new_item_embeddings, p=2, dim=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = self.bipartite_gcn_conv((item_embeddings, simple_user_embeddings), self.R_user_edge_index.flip([0]), self.R_user_edge_weight, size=(self.n_items, self.n_users))
            norm_embeddings = F.normalize(simple_user_embeddings, p=2, dim=1)
            all_embeddings_simple += [norm_embeddings]
            item_embeddings = new_item_embeddings

        # averaging the channel-specific embeddings
        user_embeddings_c1 = torch.stack(all_embeddings_c1, dim=0).sum(dim=0)
        user_embeddings_c2 = torch.stack(all_embeddings_c2, dim=0).sum(dim=0)
        user_embeddings_c3 = torch.stack(all_embeddings_c3, dim=0).sum(dim=0)
        simple_user_embeddings = torch.stack(all_embeddings_simple, dim=0).sum(dim=0)
        item_all_embeddings = torch.stack(all_embeddings_i, dim=0).sum(dim=0)

        # aggregating channel-specific embeddings
        user_all_embeddings = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
        user_all_embeddings += simple_user_embeddings / 2

        return user_all_embeddings, item_all_embeddings

    def hierarchical_self_supervision(self, user_embeddings, edge_index, edge_weight):
        def row_shuffle(embedding):
            shuffled_embeddings = embedding[torch.randperm(embedding.size(0))]
            return shuffled_embeddings
        def row_column_shuffle(embedding):
            shuffled_embeddings = embedding[:, torch.randperm(embedding.size(1))]
            shuffled_embeddings = shuffled_embeddings[torch.randperm(embedding.size(0))]
            return shuffled_embeddings
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=1)

        # For Douban, normalization is needed.
        # user_embeddings = F.normalize(user_embeddings, p=2, dim=1) 
        edge_embeddings = self.bipartite_gcn_conv((user_embeddings, user_embeddings), edge_index.flip([0]), edge_weight, size=(self.n_users, self.n_users))
        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))
        # Global MIM
        graph = torch.mean(edge_embeddings, dim=0, keepdim=True)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate self-supervised loss
        ss_loss = self.hierarchical_self_supervision(self.ss_gating_c1(user_all_embeddings), self.H_s_edge_index, self.H_s_edge_weight)
        ss_loss += self.hierarchical_self_supervision(self.ss_gating_c2(user_all_embeddings), self.H_j_edge_index, self.H_j_edge_weight)
        ss_loss += self.hierarchical_self_supervision(self.ss_gating_c3(user_all_embeddings), self.H_p_edge_index, self.H_p_edge_weight)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        loss = mf_loss + self.ssl_reg * ss_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)