# @Time   : 2022/3/22
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
GCE-GNN
################################################

Reference:
    Ziyang Wang et al. "Global Context Enhanced Graph Neural Networks for Session-based Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/CCIIPLab/GCE-GNN

"""

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender


class LocalAggregator(MessagePassing):
    def __init__(self, dim, alpha):
        super().__init__(aggr='add')
        self.edge_emb = nn.Embedding(4, dim)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        x = x_j * x_i
        a = self.edge_emb(edge_attr)
        e = (x * a).sum(dim=-1)
        e = self.leakyrelu(e)
        e = softmax(e, index, ptr, size_i)
        return e.unsqueeze(-1) * x_j


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output


class GCEGNN(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GCEGNN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.leakyrelu_alpha = config['leakyrelu_alpha']
        self.dropout_local = config['dropout_local']
        self.dropout_global = config['dropout_global']
        self.dropout_gcn = config['dropout_gcn']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.build_global_graph = config['build_global_graph']
        self.sample_num = config['sample_num']
        self.hop = config['hop']
        self.max_seq_length = dataset.field2seqlen[self.ITEM_SEQ]

        # global graph construction
        self.global_graph = None
        if self.build_global_graph:
            self.global_adj, self.global_weight = self.construct_global_graph(dataset)

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)

        # define layers and loss
        # Aggregator
        self.local_agg = LocalAggregator(self.embedding_size, self.leakyrelu_alpha)
        global_agg_list = []
        for i in range(self.hop):
            global_agg_list.append(GlobalAggregator(self.embedding_size, self.dropout_gcn))
        self.global_agg = nn.ModuleList(global_agg_list)

        self.w_1 = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=False)
        self.w_2 = nn.Linear(self.embedding_size, 1, bias=False)
        self.glu1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.glu2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.reset_parameters()
        self.other_parameter_name = ['global_adj', 'global_weight']

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _add_edge(self, graph, sid, tid):
        if tid not in graph[sid]:
            graph[sid][tid] = 0
        graph[sid][tid] += 1

    def construct_global_graph(self, dataset):
        self.logger.info('Constructing global graphs.')
        item_id_list = dataset.inter_feat['item_id_list']
        src_item_ids = item_id_list[:,:4].tolist()
        tgt_itme_id = dataset.inter_feat['item_id'].tolist()
        global_graph = [{} for _ in range(self.n_items)]
        for i in tqdm(range(len(tgt_itme_id)), desc='Converting: '):
            tid = tgt_itme_id[i]
            for sid in src_item_ids[i]:
                if sid > 0:
                    self._add_edge(global_graph, tid, sid)
                    self._add_edge(global_graph, sid, tid)
        global_adj = [[] for _ in range(self.n_items)]
        global_weight = [[] for _ in range(self.n_items)]
        for i in tqdm(range(self.n_items), desc='Sorting: '):
            sorted_out_edges = [v for v in sorted(global_graph[i].items(), reverse=True, key=lambda x: x[1])]
            global_adj[i] = [v[0] for v in sorted_out_edges[:self.sample_num]]
            global_weight[i] = [v[1] for v in sorted_out_edges[:self.sample_num]]
            if len(global_adj[i]) < self.sample_num:
                for j in range(self.sample_num - len(global_adj[i])):
                    global_adj[i].append(0)
                    global_weight[i].append(0)
        return torch.LongTensor(global_adj).to(self.device), torch.FloatTensor(global_weight).to(self.device)

    def fusion(self, hidden, mask):
        batch_size = hidden.shape[0]
        length = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:length]
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).expand(-1, length, -1)
        nh = self.w_1(torch.cat([pos_emb, hidden], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = self.w_2(nh)
        beta = beta * mask
        final_h = torch.sum(beta * hidden, 1)
        return final_h

    def forward(self, x, edge_index, edge_attr, alias_inputs, item_seq_len):
        batch_size = alias_inputs.shape[0]
        mask = alias_inputs.gt(0).unsqueeze(-1)
        h = self.item_embedding(x)

        # local
        h_local = self.local_agg(h, edge_index, edge_attr)

        # global
        item_neighbors = [F.pad(x[alias_inputs], (0, self.max_seq_length - x[alias_inputs].shape[1]), "constant", 0)]
        weight_neighbors = []
        support_size = self.max_seq_length

        for i in range(self.hop):
            item_sample_i, weight_sample_i = self.global_adj[item_neighbors[-1].view(-1)], self.global_weight[item_neighbors[-1].view(-1)]
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.item_embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = h[alias_inputs] * mask

        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask.float(), 1)

        # sum
        # sum_item_emb = torch.sum(item_emb, 1)

        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.embedding_size]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, self.max_seq_length, self.embedding_size)
        h_global = h_global[:,:alias_inputs.shape[1],:]

        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        h_local = h_local[alias_inputs]

        h_session = h_local + h_global
        h_session = self.fusion(h_session, mask)
        return h_session

    def calculate_loss(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        edge_attr = interaction['edge_attr']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, edge_attr, alias_inputs, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        x = interaction['x']
        edge_index = interaction['edge_index']
        edge_attr = interaction['edge_attr']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, edge_attr, alias_inputs, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        edge_attr = interaction['edge_attr']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, edge_attr, alias_inputs, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
