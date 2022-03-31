# @Time   : 2022/3/28
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
SRGNN
################################################

Reference:
    Zhiqiang Pan et al. "Star Graph Neural Networks for Session-based Recommendation." in CIKM 2020.

Reference code:
    https://bitbucket.org/nudtpanzq/sgnn-hn

"""

import math
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

from recbole_gnn.model.layers import SRGNNCell


def layer_norm(x):
    ave_x = torch.mean(x, -1).unsqueeze(-1)
    x = x - ave_x
    norm_x = torch.sqrt(torch.sum(x**2, -1)).unsqueeze(-1)
    y = x / norm_x
    return y


class SGNNHN(SequentialRecommender):
    r"""SGNN-HN applies a star graph neural network to model the complex transition relationship between items in an ongoing session.
        To avoid overfitting, it applies highway networks to adaptively select embeddings from item representations.
    """

    def __init__(self, config, dataset):
        super(SGNNHN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.step = config['step']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.scale = config['scale']

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.max_seq_length = dataset.field2seqlen[self.ITEM_SEQ]
        self.pos_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)

        # define layers and loss
        self.gnncell = SRGNNCell(self.embedding_size)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_three = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_four = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def att_out(self, hidden, star_node, batch):
        star_node_repeat = torch.index_select(star_node, 0, batch)
        sim = (hidden * star_node_repeat).sum(dim=-1)
        sim = softmax(sim, batch)
        att_hidden = sim.unsqueeze(-1) * hidden
        output = global_add_pool(att_hidden, batch)

        return output

    def forward(self, x, edge_index, batch, alias_inputs, item_seq_len):
        mask = alias_inputs.gt(0)
        hidden = self.item_embedding(x)

        star_node = global_mean_pool(hidden, batch)
        for i in range(self.step):
            hidden = self.gnncell(hidden, edge_index)
            star_node_repeat = torch.index_select(star_node, 0, batch)
            sim = (hidden * star_node_repeat).sum(dim=-1, keepdim=True) / math.sqrt(self.embedding_size)
            alpha = torch.sigmoid(sim)
            hidden = (1 - alpha) * hidden + alpha * star_node_repeat
            star_node = self.att_out(hidden, star_node, batch)

        seq_hidden = hidden[alias_inputs]
        bs, item_num, _ = seq_hidden.shape
        pos_emb = self.pos_embedding.weight[:item_num]
        pos_emb = pos_emb.unsqueeze(0).expand(bs, -1, -1)
        seq_hidden = seq_hidden + pos_emb

        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)
        q3 = self.linear_three(star_node).view(star_node.shape[0], 1, star_node.shape[1])

        alpha = self.linear_four(torch.sigmoid(q1 + q2 + q3))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return layer_norm(seq_output)

    def calculate_loss(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        batch = interaction['batch']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, batch, alias_inputs, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = layer_norm(self.item_embedding(pos_items))
            neg_items_emb = layer_norm(self.item_embedding(neg_items))
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1) * self.scale  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1) * self.scale  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = layer_norm(self.item_embedding.weight)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) * self.scale
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        x = interaction['x']
        edge_index = interaction['edge_index']
        batch = interaction['batch']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, batch, alias_inputs, item_seq_len)
        test_item_emb = layer_norm(self.item_embedding(test_item))
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) * self.scale  # [B]
        return scores

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        batch = interaction['batch']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, batch, alias_inputs, item_seq_len)
        test_items_emb = layer_norm(self.item_embedding.weight)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) * self.scale  # [B, n_items]
        return scores
