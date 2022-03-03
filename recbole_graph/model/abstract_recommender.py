import torch
from torch_geometric.utils import degree

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import ModelType


class GeneralGraphRecommender(GeneralRecommender):
    """This is a abstract general graph recommender. All the general graph model should implement this class.
    The base general graph recommender class provide the basic U-I graph dataset and parameters information.
    """
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralGraphRecommender, self).__init__(config, dataset)

        self.edge_index, self.edge_weight = self.get_norm_adj_mat(dataset)

    def get_norm_adj_mat(self, dataset):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = dataset.inter_feat[self.USER_ID]
        col = dataset.inter_feat[self.ITEM_ID] + self.n_users
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.n_users + self.n_items)
        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))

        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index.to(self.device), edge_weight.to(self.device)