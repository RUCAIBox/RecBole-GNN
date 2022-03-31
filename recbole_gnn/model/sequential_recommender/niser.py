# @Time   : 2022/3/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
NISER
################################################

Reference:
    Priyanka Gupta et al. "NISER: Normalized Item and Session Representations to Handle Popularity Bias." in CIKM 2019 GRLA workshop.

"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender

from recbole_gnn.model.layers import SRGNNCell


class NISER(SequentialRecommender):
    r"""NISER+ is a GNN-based model that normalizes session and item embeddings to handle popularity bias.
    """

    def __init__(self, config, dataset):
        super(NISER, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.step = config['step']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.sigma = config['sigma']
        self.max_seq_length = dataset.field2seqlen[self.ITEM_SEQ]

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        self.item_dropout = nn.Dropout(config['item_dropout'])

        # define layers and loss
        self.gnncell = SRGNNCell(self.embedding_size)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
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

    def forward(self, x, edge_index, alias_inputs, item_seq_len):
        mask = alias_inputs.gt(0)
        hidden = self.item_embedding(x)
        # Dropout in NISER+
        hidden = self.item_dropout(hidden)
        # Normalize item embeddings
        hidden = F.normalize(hidden, dim=-1)
        for i in range(self.step):
            hidden = self.gnncell(hidden, edge_index)

        seq_hidden = hidden[alias_inputs]
        batch_size = seq_hidden.shape[0]
        pos_emb = self.pos_embedding.weight[:seq_hidden.shape[1]]
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        seq_hidden = seq_hidden + pos_emb
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        # Normalize session embeddings
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def calculate_loss(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, alias_inputs, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = F.normalize(self.item_embedding(pos_items), dim=-1)
            neg_items_emb = F.normalize(self.item_embedding(neg_items), dim=-1)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(self.sigma * pos_score, self.sigma * neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = F.normalize(self.item_embedding.weight, dim=-1)
            logits = self.sigma * torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, alias_inputs, item_seq_len)
        test_item_emb = F.normalize(self.item_embedding(test_item), dim=-1)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, alias_inputs, item_seq_len)
        test_items_emb = F.normalize(self.item_embedding.weight, dim=-1)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
