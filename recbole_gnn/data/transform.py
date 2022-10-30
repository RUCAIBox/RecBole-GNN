from logging import getLogger
import torch
from torch.nn.utils.rnn import pad_sequence
from recbole.data.interaction import Interaction


def gnn_construct_transform(config):
    if config['gnn_transform'] is None:
        raise ValueError('config["gnn_transform"] is None but trying to construct transform.')
    str2transform = {
        'sess_graph': SessionGraph,
    }
    return str2transform[config['gnn_transform']](config)


class SessionGraph:
    def __init__(self, config):
        self.logger = getLogger()
        self.logger.info('SessionGraph Transform in DataLoader.')

    def __call__(self, dataset, interaction):
        graph_objs = dataset.graph_objs
        index = interaction['graph_idx']
        graph_batch = {
            k: [graph_objs[k][_.item()] for _ in index]
            for k in graph_objs
        }
        graph_batch['batch'] = []

        tot_node_num = torch.ones([1], dtype=torch.long)
        for i in range(index.shape[0]):
            for k in graph_batch:
                if 'edge_index' in k:
                    graph_batch[k][i] = graph_batch[k][i] + tot_node_num
            if 'alias_inputs' in graph_batch:
                graph_batch['alias_inputs'][i] = graph_batch['alias_inputs'][i] + tot_node_num
            graph_batch['batch'].append(torch.full_like(graph_batch['x'][i], i))
            tot_node_num += graph_batch['x'][i].shape[0]

        if hasattr(dataset, 'node_attr'):
            node_attr = ['batch'] + dataset.node_attr
        else:
            node_attr = ['x', 'batch']
        for k in node_attr:
            graph_batch[k] = [torch.zeros([1], dtype=graph_batch[k][-1].dtype)] + graph_batch[k]

        for k in graph_batch:
            if k == 'alias_inputs':
                graph_batch[k] = pad_sequence(graph_batch[k], batch_first=True)
            else:
                graph_batch[k] = torch.cat(graph_batch[k], dim=-1)

        interaction.update(Interaction(graph_batch))
        return interaction
