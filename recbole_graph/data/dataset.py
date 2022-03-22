import os
import torch
import numpy as np

from tqdm import tqdm
from torch_geometric.utils import degree

from recbole.data.dataset import SequentialDataset
from recbole.data.dataset import Dataset
from recbole.utils import set_color, FeatureSource


class SessionGraphDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def session_graph_construction(self):
        # Default session graph dataset follows the graph construction operator like SR-GNN.
        self.logger.info('Constructing session graphs.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        x = []
        edge_index = []
        alias_inputs = []

        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)
            # No repeat click
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': x,
            'edge_index': edge_index,
            'alias_inputs': alias_inputs
        }

    def build(self):
        datasets = super().build()
        for dataset in datasets:
            dataset.session_graph_construction()
        return datasets


class LESSRDataset(SessionGraphDataset):
    def session_graph_construction(self):
        self.logger.info('Constructing LESSR session graphs.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]

        empty_edge = torch.stack([torch.LongTensor([]), torch.LongTensor([])])

        x = []
        edge_index_EOP = []
        edge_index_shortcut = []
        is_last = []

        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]])
            edge_index_EOP.append(edge)
            last = torch.zeros_like(seq, dtype=torch.bool)
            last[alias_seq[-1]] = True
            is_last.append(last)
            sub_edges = []
            for j in range(1, item_seq_len[i]):
                sub_edges.append(torch.stack([alias_seq[:-j], alias_seq[j:]]))
            shortcut_edge = torch.cat(sub_edges, dim=-1).unique(dim=-1) if len(sub_edges) > 0 else empty_edge
            edge_index_shortcut.append(shortcut_edge)

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': x,
            'edge_index_EOP': edge_index_EOP,
            'edge_index_shortcut': edge_index_shortcut,
            'is_last': is_last
        }
        self.node_attr = ['x', 'is_last']


class GCEGNNDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def reverse_session(self):
        self.logger.info('Reversing sessions.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        for i in tqdm(range(item_seq.shape[0])):
            item_seq[i,:item_seq_len[i]] = item_seq[i,:item_seq_len[i]].flip(dims=[0])

    def bidirectional_edge(self, edge_index):
        seq_len = edge_index.shape[1]
        ed = edge_index.T
        ed2 = edge_index.T.flip(dims=[1])
        idc = ed.unsqueeze(1).expand(-1, seq_len, 2) == ed2.unsqueeze(0).expand(seq_len, -1, 2)
        return torch.logical_and(idc[:,:,0], idc[:,:,1]).any(dim=-1)

    def session_graph_construction(self):
        self.logger.info('Constructing session graphs.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        x = []
        edge_index = []
        edge_attr = []
        alias_inputs = []

        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)

            edge_index_backward = torch.stack([alias_seq[:-1], alias_seq[1:]])
            edge_attr_backward = torch.where(self.bidirectional_edge(edge_index_backward), 3, 1)
            edge_backward = torch.cat([edge_index_backward, edge_attr_backward.unsqueeze(0)], dim=0)

            edge_index_forward = torch.stack([alias_seq[1:], alias_seq[:-1]])
            edge_attr_forward = torch.where(self.bidirectional_edge(edge_index_forward), 3, 2)
            edge_forward = torch.cat([edge_index_forward, edge_attr_forward.unsqueeze(0)], dim=0)

            edge_index_selfloop = torch.stack([alias_seq, alias_seq])
            edge_selfloop = torch.cat([edge_index_selfloop, torch.zeros([1, edge_index_selfloop.shape[1]])], dim=0)

            edge = torch.cat([edge_backward, edge_forward, edge_selfloop], dim=-1).long()
            edge = edge.unique(dim=-1)

            cur_edge_index = edge[:2]
            cur_edge_attr = edge[2]
            edge_index.append(cur_edge_index)
            edge_attr.append(cur_edge_attr)

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'alias_inputs': alias_inputs
        }

    def build(self):
        datasets = super().build()
        for dataset in datasets:
            dataset.reverse_session()
            dataset.session_graph_construction()
        return datasets


class SocialDataset(Dataset):
    """:class:`SocialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and load ``.net``.

    All users in ``.inter`` and ``.net`` are remapped into the same ID sections.
    Users that only exist in social network will be filtered.

    It also provides several interfaces to transfer ``.net`` features into coo sparse matrix,
    csr sparse matrix, :class:`DGL.Graph` or :class:`PyG.Data`.

    Attributes:
        net_src_field (str): The same as ``config['NET_SOURCE_ID_FIELD']``.

        net_tgt_field (str): The same as ``config['NET_TARGET_ID_FIELD']``.

        net_feat (pandas.DataFrame): Internal data structure stores the users' social network relations.
            It's loaded from file ``.net``.
    """
    def __init__(self, config):
        super().__init__(config)
    
    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.net_src_field = self.config['NET_SOURCE_ID_FIELD']
        self.net_tgt_field = self.config['NET_TARGET_ID_FIELD']
        self.filter_net_by_inter = self.config['filter_net_by_inter']
        self._check_field('net_src_field', 'net_tgt_field')

        self.logger.debug(set_color('net_src_field', 'blue') + f': {self.net_src_field}')
        self.logger.debug(set_color('net_tgt_field', 'blue') + f': {self.net_tgt_field}')

    def _data_filtering(self):
        super()._data_filtering()
        if self.filter_net_by_inter:
            self._filter_net_by_inter()

    def _filter_net_by_inter(self):
        """Filter users in ``net_feat`` that don't occur in interactions.
        """
        inter_uids = set(self.inter_feat[self.uid_field])
        self.net_feat.drop(self.net_feat.index[~self.net_feat[self.net_src_field].isin(inter_uids)], inplace=True)
        self.net_feat.drop(self.net_feat.index[~self.net_feat[self.net_tgt_field].isin(inter_uids)], inplace=True)

    def _load_data(self, token, dataset_path):
        super()._load_data(token, dataset_path)
        self.net_feat = self._load_net(self.dataset_name, self.dataset_path)

    @property
    def net_num(self):
        """Get the number of social network records.

        Returns:
            int: Number of social network records.
        """
        return len(self.net_feat)

    def __str__(self):
        info = [
            super().__str__(),
            set_color('The number of social network relations', 'blue') + f': {self.net_num}'
        ]  # yapf: disable
        return '\n'.join(info)

    def _build_feat_name_list(self):
        feat_name_list = super()._build_feat_name_list()
        if self.net_feat is not None:
            feat_name_list.append('net_feat')
        return feat_name_list

    def _load_net(self, token, dataset_path):
        self.logger.debug(set_color(f'Loading social network from [{dataset_path}].', 'green'))
        net_path = os.path.join(dataset_path, f'{token}.net')
        if not os.path.isfile(net_path):
            raise ValueError(f'[{token}.net] not found in [{dataset_path}].')
        df = self._load_feat(net_path, FeatureSource.NET)
        self._check_net(df)
        return df

    def _check_net(self, net):
        net_warn_message = 'net data requires field [{}]'
        assert self.net_src_field in net, net_warn_message.format(self.net_src_field)
        assert self.net_tgt_field in net, net_warn_message.format(self.net_tgt_field)

    def _init_alias(self):
        """Add :attr:`alias_of_user_id`.
        """
        self._set_alias('user_id', [self.uid_field, self.net_src_field, self.net_tgt_field])
        self._set_alias('item_id', [self.iid_field])

        for alias_name_1, alias_1 in self.alias.items():
            for alias_name_2, alias_2 in self.alias.items():
                if alias_name_1 != alias_name_2:
                    intersect = np.intersect1d(alias_1, alias_2, assume_unique=True)
                    if len(intersect) > 0:
                        raise ValueError(
                            f'`alias_of_{alias_name_1}` and `alias_of_{alias_name_2}` '
                            f'should not have the same field {list(intersect)}.'
                        )

        self._rest_fields = self.token_like_fields
        for alias_name, alias in self.alias.items():
            isin = np.isin(alias, self._rest_fields, assume_unique=True)
            if isin.all() is False:
                raise ValueError(
                    f'`alias_of_{alias_name}` should not contain '
                    f'non-token-like field {list(alias[~isin])}.'
                )
            self._rest_fields = np.setdiff1d(self._rest_fields, alias, assume_unique=True)

    def get_norm_adj_mat(self, row_norm=False):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        if row_norm:
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)
            edge_weight = norm_deg[edge_index[0]]
        else:
            norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
            edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def get_norm_net_adj_mat(self, row_norm=False):
        r"""Get the normalized socail matrix of users and users.
        Construct the square matrix from the social network data and 
        normalize it using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized social network matrix in Tensor.
        """

        row = self.net_feat[self.net_src_field]
        col = self.net_feat[self.net_tgt_field]
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num)

        if row_norm:
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)
            edge_weight = norm_deg[edge_index[0]]
        else:
            norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
            edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight