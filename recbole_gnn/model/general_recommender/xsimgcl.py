# -*- coding: utf-8 -*-
r"""
XSimGCL
################################################
Reference:
    Junliang Yu, Xin Xia, Tong Chen, Lizhen Cui, Nguyen Quoc Viet Hung, Hongzhi Yin. "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation" in TKDE 2023.

Reference code:
    https://github.com/Coder-Yu/SELFRec/blob/main/model/graph/XSimGCL.py
"""


import torch
import torch.nn.functional as F

from recbole_gnn.model.general_recommender import LightGCN


class XSimGCL(LightGCN):
    def __init__(self, config, dataset):
        super(XSimGCL, self).__init__(config, dataset)

        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']
        self.layer_cl = config['layer_cl']

    def forward(self, perturbed=False):
        all_embs = self.get_ego_embeddings()
        all_embs_cl = all_embs
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            if perturbed:
                random_noise = torch.rand_like(all_embs, device=all_embs.device)
                all_embs = all_embs + torch.sign(all_embs) * F.normalize(random_noise, dim=-1) * self.eps
            embeddings_list.append(all_embs)
            if layer_idx == self.layer_cl - 1:
                all_embs_cl = all_embs
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embs_cl, [self.n_users, self.n_items])
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl = self.forward(perturbed=True)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        user = torch.unique(interaction[self.USER_ID])
        pos_item = torch.unique(interaction[self.ITEM_ID])

        # calculate CL Loss
        user_cl_loss = self.calculate_cl_loss(user_all_embeddings[user], user_all_embeddings_cl[user])
        item_cl_loss = self.calculate_cl_loss(item_all_embeddings[pos_item], item_all_embeddings_cl[pos_item])

        return mf_loss, self.reg_weight * reg_loss, self.cl_rate * (user_cl_loss + item_cl_loss)
