# r"""
# DiretAU
# ################################################
# Reference:
#     Chenyang Wang et al. "Towards Representation Alignment and Uniformity in Collaborative Filtering." in KDD 2022.

# Reference code:
#     https://github.com/THUwangcy/DirectAU
# """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.general_recommender import BPR
from recbole_gnn.model.general_recommender import LightGCN

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender


class DirectAU(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DirectAU, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.gamma = config['gamma']
        self.encoder_name = config['encoder']

        # define encoder
        if self.encoder_name == 'MF':
            self.encoder = MFEncoder(config, dataset)
        elif self.encoder_name == 'LightGCN':
            self.encoder = LGCNEncoder(config, dataset)
        else:
            raise ValueError('Non-implemented Encoder.')

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user_e, item_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        return align, uniform

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.encoder_name == 'LightGCN':
            if self.restore_user_e is None or self.restore_item_e is None:
                self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
            user_e = self.restore_user_e[user]
            all_item_e = self.restore_item_e
        else:
            user_e = self.encoder.user_embedding(user)
            all_item_e = self.encoder.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)


class MFEncoder(BPR):
    def __init__(self, config, dataset):
        super(MFEncoder, self).__init__(config, dataset)

    def forward(self, user_id, item_id):
        return super().forward(user_id, item_id)

    def get_all_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings, item_embeddings


class LGCNEncoder(LightGCN):
    def __init__(self, config, dataset):
        super(LGCNEncoder, self).__init__(config, dataset)

    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed

    def get_all_embeddings(self):
        return super().forward()
