# @Time   : 2022/3/22
# @Author : Lanling Xu
# @Email  : xulanling_sherry@163.com

r"""
GraphRec
################################################
Reference:
    Wenqi Fan et al. "Graph Neural Networks for Social Recommendation." in WWW 2019.

Reference code:
    https://github.com/wenqifan03/GraphRec-WWW19
"""

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType

from recbole_graph.model.abstract_recommender import SocialRecommender
from recbole_graph.model.layers import LightGCNConv


class Attention(nn.Module):
    r"""Attention mechanism in aggregators for attention score calculation.
    """
    def __init__(self, emb_dim):
        super(Attention, self).__init__()
        self.att1 = nn.Linear(emb_dim * 2, emb_dim)
        self.att2 = nn.Linear(emb_dim, emb_dim)
        self.att3 = nn.Linear(emb_dim, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x_j, x_i):
        x = torch.cat((x_j, x_i), dim=1)
        x = self.dropout(self.activation(self.att1(x)))
        x = self.dropout(self.activation(self.att2(x)))
        x = self.att3(x)
        return x


class UIAggregator(MessagePassing):
    r"""UIAggregator is to capture interactions and opinions in the user-item graph.
    """
    def __init__(self, opinion_emb, emb_dim):
        super().__init__(aggr='add')
        self.opinion_emb = opinion_emb
        self.ln1 = nn.Linear(emb_dim * 2, emb_dim)
        self.ln2 = nn.Linear(emb_dim, emb_dim)
        self.activation = nn.ReLU()
        self.att = Attention(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        r = self.opinion_emb(edge_attr)
        x = torch.cat((x_j, r), dim=1)
        x = self.activation(self.ln1(x))
        x_a_j = self.activation(self.ln2(x))
        e = self.att(x_a_j, x_i)
        e = softmax(e, index, ptr, size_i)
        return e * x_a_j


class SocialAggregator(MessagePassing):
    r"""SocialAggregator is to capture social network connections in the user-user graph.
    """
    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.att = Attention(emb_dim)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i, index, ptr, size_i):
        e = self.att(x_j, x_i)
        e = softmax(e, index, ptr, size_i)
        return e * x_j


class GraphRec(SocialRecommender):
    r"""GraphRec is a graph neural network framework for social recommendations, providing a 
    principled approach to jointly capture interactions and opinions in the user-item graph.
    In particular, GraphRec coherently models two graphs and heterogeneous strengths.
    
    We implement the model following the original author with a pointwise training mode.
    For generalization and comparability, we provide two input forms: explicit rating score 
    and implicit 0/1 interaction. If ``inter_matrix_type='rating'``, `rating_list` is the 
    list of different score values, which is used to express users' opinions on items. If ``inter_matrix_type='01'``, `rating_list` can only be ``[1]``.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(GraphRec, self).__init__(config, dataset)

        # load dataset info
        self.edge_index, _ = dataset.get_norm_adj_mat(row_norm=True)
        self.edge_index = self.edge_index.to(self.device)

        self.net_edge_index, _ = dataset.get_norm_net_adj_mat(row_norm=True)
        self.net_edge_index = self.net_edge_index.to(self.device)

        # load parameters info
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.mlp_layer_num = config['mlp_layer_num']
        self.norm_momentum = config['norm_momentum']
        self.rating_list = config['rating_list']
        self.inter_matrix_type = config['inter_matrix_type']

        self.edge_attr = self.get_edge_attr(dataset)
        self.n_opinions = len(self.rating_list)

        # define ego embeddings and loss
        self.user_embedding = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.item_embedding = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)
        self.opinion_embedding = nn.Embedding(num_embeddings=self.n_opinions, embedding_dim=self.embedding_size)
        self.loss_fct = nn.MSELoss()

        # define aggregator for neighborhood propagation and aggregation
        self.u_aggregator = UIAggregator(self.opinion_embedding, self.embedding_size)
        self.i_aggregator = UIAggregator(self.opinion_embedding, self.embedding_size)
        self.social_aggregator = SocialAggregator(self.embedding_size)

        # define concatenation layers for combining embedding itself and its aggregated neighbors
        self.u_agg_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.i_agg_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.social_agg_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.activation = nn.ReLU()

        # define mlp layers for score calculation
        self.u_encode_layer_dims = [self.embedding_size] * self.mlp_layer_num
        self.i_encode_layer_dims = [self.embedding_size] * self.mlp_layer_num
        self.ui_encode_layer_dims = [self.embedding_size * 2, self.embedding_size, self.hidden_size, 1]

        self.u_encoder = self.mlp_layers(self.u_encode_layer_dims)
        self.i_encoder = self.mlp_layers(self.i_encode_layer_dims)
        self.ui_encoder = self.mlp_layers(self.ui_encode_layer_dims)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_edge_attr(self, dataset):
        r"""Get the normalized indexes of ratings in opinion embeddings.
        """
        if self.inter_matrix_type == '01':
            edge_attr = torch.zeros(self.edge_index.size(1))
        elif self.inter_matrix_type == 'rating':
            ratings = dataset.inter_feat[self.RATING]
            opinion_dict = {value: key for key, value in enumerate(self.rating_list)}
            edge_attr = torch.tensor([opinion_dict[x.item()] for x in ratings])
        else:
            raise NotImplementedError("Make sure 'inter_matrix_type' in ['01', 'rating']!")
        return edge_attr.int().to(self.device)

    def forward(self):
        # get ego embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        # Item Aggregation in User Modeling
        user_all_ui_embeddings = self.u_aggregator(all_embeddings, self.edge_index, self.edge_attr)
        user_ui_embeddings, _ = torch.split(user_all_ui_embeddings, [self.n_users, self.n_items])
        user_ui_embeddings = self.activation(self.u_agg_linear(torch.cat([user_embeddings, user_ui_embeddings], dim=1)))

        # Social Aggregation in User Modeling
        user_social_embeddings = self.social_aggregator(user_embeddings, self.net_edge_index)

        # Learning User Latent Factor
        user_all_embeddings = self.activation(self.social_agg_linear(torch.cat([user_ui_embeddings, user_social_embeddings], dim=1)))

        # User Aggregation in Item Modeling
        item_all_ui_embeddings = self.i_aggregator(all_embeddings, self.edge_index, self.edge_attr)
        _, item_all_embeddings = torch.split(item_all_ui_embeddings, [self.n_users, self.n_items])
        item_all_embeddings = self.activation(self.i_agg_linear(torch.cat([item_embeddings, item_all_embeddings], dim=1)))

        return user_all_embeddings, item_all_embeddings

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.BatchNorm1d(d_out, momentum=self.norm_momentum))
                mlp_modules.append(nn.ReLU())
                mlp_modules.append(nn.Dropout())
        return nn.Sequential(*mlp_modules)

    def calculate_score(self, user_e, item_e):
        x_u = self.u_encoder(user_e)
        x_i = self.i_encoder(item_e)

        x_ui = torch.cat((x_u, x_i), dim=1)
        score = self.ui_encoder(x_ui)

        return score.squeeze()

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.inter_matrix_type == '01':
            label = interaction[self.LABEL]
        elif self.inter_matrix_type == 'rating':
            label = interaction[self.RATING] * interaction[self.LABEL]
        else:
            raise NotImplementedError("Make sure 'inter_matrix_type' in ['01', 'rating']!")

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        scores = self.calculate_score(u_embeddings, i_embeddings)
        loss = self.loss_fct(scores, label)
        
        return loss

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_item_e[item]
        scores = self.calculate_score(u_embeddings, i_embeddings)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        scores = []
        for u_e in u_embeddings:
            score = self.calculate_score(u_e.unsqueeze(dim=0).repeat(self.restore_item_e.size(0), 1), self.restore_item_e)
            scores.append(score)
        return torch.stack(scores, dim=0).view(-1)