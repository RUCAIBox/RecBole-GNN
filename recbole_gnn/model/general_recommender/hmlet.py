# @Time   : 2022/3/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
HMLET
################################################
Reference:
    Taeyong Kong et al. "Linear, or Non-Linear, That is the Question!." in WSDM 2022.

Reference code:
    https://github.com/qbxlvnf11/HMLET
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.layers import activation_layer
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class Gating_Net(nn.Module):
    def __init__(self, embedding_dim, mlp_dims, dropout_p):
        super(Gating_Net, self).__init__()
        self.embedding_dim = embedding_dim

        fc_layers = []
        for i in range(len(mlp_dims)):
            if i == 0:
                fc = nn.Linear(embedding_dim*2, mlp_dims[i])
                fc_layers.append(fc)
            else:
                fc = nn.Linear(mlp_dims[i-1], mlp_dims[i])
                fc_layers.append(fc)
            if i != len(mlp_dims) - 1:
                fc_layers.append(nn.BatchNorm1d(mlp_dims[i]))
                fc_layers.append(nn.Dropout(p=dropout_p))
                fc_layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*fc_layers)

    def gumbel_softmax(self, logits, temperature, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature) ## (0.6, 0.2, 0.1,..., 0.11)
        if hard:
            k = logits.size(1) # k is numb of classes
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        return y

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        noise = self.sample_gumbel(logits)
        y = (logits + noise) / temperature
        return F.softmax(y, dim=1)

    def sample_gumbel(self, logits):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(logits.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return torch.Tensor(noise.float()).to(logits.device)

    def forward(self, feature, temperature, hard):
        x = self.mlp(feature)
        out = self.gumbel_softmax(x, temperature, hard)
        out_value = out.unsqueeze(2)
        gating_out = out_value.repeat(1, 1, self.embedding_dim)
        return gating_out


class HMLET(GeneralGraphRecommender):
    r"""HMLET combines both linear and non-linear propagation layers for general recommendation and yields better performance.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HMLET, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
        self.gate_layer_ids = config['gate_layer_ids']  # list type: layer ids for non-linear gating
        self.gating_mlp_dims = config['gating_mlp_dims']  # list type: list of mlp dimensions in gating module
        self.dropout_ratio = config['dropout_ratio']  # dropout ratio for mlp in gating module
        self.gum_temp = config['ori_temp']
        self.logger.info(f'Model initialization, gumbel softmax temperature: {self.gum_temp}')

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.activation = nn.ELU() if config['activation_function'] == 'elu' else activation_layer(config['activation_function'])
        self.gating_nets = nn.ModuleList([
            Gating_Net(self.latent_dim, self.gating_mlp_dims, self.dropout_ratio) for _ in range(len(self.gate_layer_ids))
        ])

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e', 'gum_temp']

        for gating in self.gating_nets:
            self._gating_freeze(gating, False)

    def _gating_freeze(self, model, freeze_flag):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = freeze_flag

    def __choosing_one(self, features, gumbel_out):
        feature = torch.sum(torch.mul(features, gumbel_out), dim=1)  # batch x embedding_dim (or batch x embedding_dim x layer_num)
        return feature

    def __where(self, idx, lst):
        for i in range(len(lst)):
            if lst[i] == idx:
                return i
        raise ValueError(f'{idx} not in {lst}.')

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        non_lin_emb_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            linear_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            if layer_idx not in self.gate_layer_ids:
                all_embeddings = linear_embeddings
            else:
                non_lin_id = self.__where(layer_idx, self.gate_layer_ids)
                last_non_lin_emb = non_lin_emb_list[non_lin_id]
                non_lin_embeddings = self.activation(self.gcn_conv(last_non_lin_emb, self.edge_index, self.edge_weight))
                stack_embeddings = torch.stack([linear_embeddings, non_lin_embeddings], dim=1)
                concat_embeddings = torch.cat((linear_embeddings, non_lin_embeddings), dim=-1)
                gumbel_out = self.gating_nets[non_lin_id](concat_embeddings, self.gum_temp, not self.training)
                all_embeddings = self.__choosing_one(stack_embeddings, gumbel_out)
                non_lin_emb_list.append(all_embeddings)
            embeddings_list.append(all_embeddings)
        hmlet_all_embeddings = torch.stack(embeddings_list, dim=1)
        hmlet_all_embeddings = torch.mean(hmlet_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(hmlet_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

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

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)
        loss = mf_loss + self.reg_weight * reg_loss

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