import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
try:
    from torch_sparse import SparseTensor
    is_sparse = True
except ImportError:
    is_sparse = False

from recbole.data.dataset import SequentialDataset
from recbole.data.dataset import Dataset as RecBoleDataset
from recbole.utils import set_color, FeatureSource

import recbole
import pickle
from recbole.utils import ensure_dir


class GeneralGraphDataset(RecBoleDataset):
    def __init__(self, config):
        super().__init__(config)

    if recbole.__version__ == "1.1.1":

        def save(self):
            """Saving this :class:`Dataset` object to :attr:`config['checkpoint_dir']`."""
            save_dir = self.config["checkpoint_dir"]
            ensure_dir(save_dir)
            file = os.path.join(save_dir, f'{self.config["dataset"]}-{self.__class__.__name__}.pth')
            self.logger.info(
                set_color("Saving filtered dataset into ", "pink") + f"[{file}]"
            )
            with open(file, "wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes):
        adj = SparseTensor(row=edge_index[0],
                           col=edge_index[1],
                           value=edge_weight,
                           sparse_sizes=(m_num_nodes, n_num_nodes))
        return adj.t()

    def get_norm_adj_mat(self, enable_sparse=False):
        self.is_sparse = is_sparse
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
        edge_weight = torch.ones(edge_index.size(1))
        num_nodes = self.user_num + self.item_num

        if enable_sparse:
            if not is_sparse:
                self.logger.warning(
                    "Import `torch_sparse` error, please install corrsponding version of `torch_sparse`. Now we will use dense edge_index instead of SparseTensor in dataset.")
            else:
                adj_t = self.edge_index_to_adj_t(edge_index, edge_weight, num_nodes, num_nodes)
                adj_t = gcn_norm(adj_t, None, num_nodes, add_self_loops=False)
                return adj_t, None

        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=False)

        return edge_index, edge_weight

    def get_bipartite_inter_mat(self, row='user', row_norm=True):
        r"""Get the row-normalized bipartite interaction matrix of users and items.
        """
        if row == 'user':
            row_field, col_field = self.uid_field, self.iid_field
        else:
            row_field, col_field = self.iid_field, self.uid_field

        row = self.inter_feat[row_field]
        col = self.inter_feat[col_field]
        edge_index = torch.stack([row, col])

        if row_norm:
            deg = degree(edge_index[0], self.num(row_field))
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)
            edge_weight = norm_deg[edge_index[0]]
        else:
            row_deg = degree(edge_index[0], self.num(row_field))
            col_deg = degree(edge_index[1], self.num(col_field))

            row_norm_deg = 1. / torch.sqrt(torch.where(row_deg == 0, torch.ones([1]), row_deg))
            col_norm_deg = 1. / torch.sqrt(torch.where(col_deg == 0, torch.ones([1]), col_deg))

            edge_weight = row_norm_deg[edge_index[0]] * col_norm_deg[edge_index[1]]

        return edge_index, edge_weight


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


class MultiBehaviorDataset(SessionGraphDataset):

    def session_graph_construction(self):
        self.logger.info('Constructing multi-behavior session graphs.')
        self.item_behavior_list_field = self.config['ITEM_BEHAVIOR_LIST_FIELD']
        self.behavior_id_field = self.config['BEHAVIOR_ID_FIELD']
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        if self.item_behavior_list_field == None or self.behavior_id_field == None:
            # To be compatible with existing datasets
            item_behavior_seq = torch.tensor([0] * len(item_seq))
            self.behavior_id_field = 'behavior_id'
            self.field2id_token[self.behavior_id_field] = {0: 'interaction'}
        else:
            item_behavior_seq = self.inter_feat[self.item_list_length_field]

        edge_index = []
        alias_inputs = []
        behaviors = torch.unique(item_behavior_seq)
        x = {}
        for behavior in behaviors:
            x[behavior.item()] = []

        behavior_seqs = list(torch.chunk(item_behavior_seq, item_seq.shape[0]))
        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            bseq = behavior_seqs[i]
            for behavior in behaviors:
                bidx = torch.where(bseq == behavior)
                subseq = torch.index_select(seq, 0, bidx[0])
                subseq, _ = torch.unique(subseq, return_inverse=True)
                x[behavior.item()].append(subseq)

            seq, idx = torch.unique(seq, return_inverse=True)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)
            # No repeat click
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)

        nx = {}
        for k, v in x.items():
            behavior_name = self.id2token(self.behavior_id_field, k)
            nx[behavior_name] = v

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': nx,
            'edge_index': edge_index,
            'alias_inputs': alias_inputs
        }


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
            item_seq[i, :item_seq_len[i]] = item_seq[i, :item_seq_len[i]].flip(dims=[0])

    def bidirectional_edge(self, edge_index):
        seq_len = edge_index.shape[1]
        ed = edge_index.T
        ed2 = edge_index.T.flip(dims=[1])
        idc = ed.unsqueeze(1).expand(-1, seq_len, 2) == ed2.unsqueeze(0).expand(seq_len, -1, 2)
        return torch.logical_and(idc[:, :, 0], idc[:, :, 1]).any(dim=-1)

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


class SocialDataset(GeneralGraphDataset):
    """:class:`SocialDataset` is based on :class:`~recbole_gnn.data.dataset.GeneralGraphDataset`,
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
        self.undirected_net = self.config['undirected_net']
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
        if self.undirected_net:
            row = df[self.net_src_field]
            col = df[self.net_tgt_field]
            df_net_src = pd.concat([row, col], axis=0)
            df_net_tgt = pd.concat([col, row], axis=0)
            df_net_src.name = self.net_src_field
            df_net_tgt.name = self.net_tgt_field
            df = pd.concat([df_net_src, df_net_tgt], axis=1)
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
        edge_index = torch.stack([row, col])

        deg = degree(edge_index[0], self.user_num)

        if row_norm:
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)
            edge_weight = norm_deg[edge_index[0]]
        else:
            norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
            edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def net_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe social relations between user_id and user_id.

        Sparse matrix has shape (user_num, user_num).

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        return self._create_sparse_matrix(self.net_feat, self.net_src_field, self.net_tgt_field, form, value_field)
