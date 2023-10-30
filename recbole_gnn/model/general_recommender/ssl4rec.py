r"""
SSL4REC
################################################
Reference:
    Tiansheng Yao et al. "Self-supervised Learning for Large-scale Item Recommendations." in CIKM 2021.

Reference code:
    https://github.com/Coder-Yu/SELFRec/model/graph/SSL4Rec.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.loss import EmbLoss
from recbole.utils import InputType

from recbole.model.init import xavier_uniform_initialization
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender


class SSL4REC(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SSL4REC, self).__init__(config, dataset)

        # load parameters info
        self.tau = config["tau"]
        self.reg_weight = config["reg_weight"]
        self.cl_rate = config["ssl_weight"]
        self.require_pow = config["require_pow"]

        self.reg_loss = EmbLoss()

        self.encoder = DNN_Encoder(config, dataset)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def forward(self, user, item):
        user_e, item_e = self.encoder(user, item)
        return user_e, item_e

    def calculate_batch_softmax_loss(self, user_emb, item_emb, temperature):
        user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
        pos_score = (user_emb * item_emb).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        loss = -torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(loss)

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]

        user_embeddings, item_embeddings = self.forward(user, pos_item)

        rec_loss = self.calculate_batch_softmax_loss(user_embeddings, item_embeddings, self.tau)
        cl_loss = self.encoder.calculate_cl_loss(pos_item)
        reg_loss = self.reg_loss(user_embeddings, item_embeddings, require_pow=self.require_pow)

        loss = rec_loss + self.cl_rate * cl_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_embeddings, item_embeddings = self.forward(user, item)

        u_embeddings = user_embeddings[user]
        i_embeddings = item_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(torch.arange(
                self.n_users, device=self.device), torch.arange(self.n_items, device=self.device))
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class DNN_Encoder(nn.Module):
    def __init__(self, config, dataset):
        super(DNN_Encoder, self).__init__()

        self.emb_size = config["embedding_size"]
        self.drop_ratio = config["drop_ratio"]
        self.tau = config["tau"]

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        self.user_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.item_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(self.drop_ratio)

        self.initial_user_emb = nn.Embedding(self.n_users, self.emb_size)
        self.initial_item_emb = nn.Embedding(self.n_items, self.emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.initial_user_emb.weight)
        nn.init.xavier_uniform_(self.initial_item_emb.weight)

    def forward(self, q, x):
        q_emb = self.initial_user_emb(q)
        i_emb = self.initial_item_emb(x)

        q_emb = self.user_tower(q_emb)
        i_emb = self.item_tower(i_emb)

        return q_emb, i_emb

    def item_encoding(self, x):
        i_emb = self.initial_item_emb(x)
        i1_emb = self.dropout(i_emb)
        i2_emb = self.dropout(i_emb)

        i1_emb = self.item_tower(i1_emb)
        i2_emb = self.item_tower(i2_emb)

        return i1_emb, i2_emb

    def calculate_cl_loss(self, idx):
        x1, x2 = self.item_encoding(idx)
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()
