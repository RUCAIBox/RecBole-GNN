# @Time   : 2022/3/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
GCSAN
################################################

Reference:
    Chengfeng Xu et al. "Graph Contextualized Self-Attention Network for Session-based Recommendation." in IJCAI 2019.

"""

import torch
from torch import nn
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender

from recbole_gnn.model.layers import SRGNNCell


class GCSAN(SequentialRecommender):
    r"""GCSAN captures rich local dependencies via graph neural network,
     and learns long-range dependencies by applying the self-attention mechanism.
     
    Note:

        In the original paper, the attention mechanism in the self-attention layer is a single head,
        for the reusability of the project code, we use a unified transformer component.
        According to the experimental results, we only applied regularization to embedding.
    """

    def __init__(self, config, dataset):
        super(GCSAN, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.step = config['step']
        self.device = config['device']
        self.weight = config['weight']
        self.reg_weight = config['reg_weight']
        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        # define layers and loss
        self.gnncell = SRGNNCell(self.hidden_size)
        self.self_attention = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.reg_loss = EmbLoss()
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, x, edge_index, alias_inputs, item_seq_len):
        hidden = self.item_embedding(x)
        for i in range(self.step):
            hidden = self.gnncell(hidden, edge_index)

        seq_hidden = hidden[alias_inputs]
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)

        attention_mask = self.get_attention_mask(alias_inputs)
        outputs = self.self_attention(seq_hidden, attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        at = self.gather_indexes(output, item_seq_len - 1)
        seq_output = self.weight * at + (1 - self.weight) * ht
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
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        reg_loss = self.reg_loss(self.item_embedding.weight)
        total_loss = loss + self.reg_weight * reg_loss
        return total_loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, alias_inputs, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, alias_inputs, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
