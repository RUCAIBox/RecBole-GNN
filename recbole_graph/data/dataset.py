from tqdm import tqdm
import torch
from recbole.data.dataset import SequentialDataset


class SessionGraphDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def session_graph_construction(self):
        raise NotImplementedError('Graph construction method needs to be implemented.')

    def build(self):
        datasets = super().build()
        for dataset in datasets:
            dataset.session_graph_construction()
        return datasets


class SRGNNDataset(SessionGraphDataset):
    def session_graph_construction(self):
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


class GCSANDataset(SRGNNDataset):
    pass


class NISERDataset(SRGNNDataset):
    pass


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
