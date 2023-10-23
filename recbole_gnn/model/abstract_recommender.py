from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import ModelType as RecBoleModelType

from recbole_gnn.utils import ModelType


class GeneralGraphRecommender(GeneralRecommender):
    """This is an abstract general graph recommender. All the general graph models should implement in this class.
    The base general graph recommender class provide the basic U-I graph dataset and parameters information.
    """
    type = RecBoleModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralGraphRecommender, self).__init__(config, dataset)
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat(enable_sparse=config["enable_sparse"])
        self.use_sparse = config["enable_sparse"] and dataset.is_sparse
        if self.use_sparse:
            self.edge_index, self.edge_weight = self.edge_index.to(self.device), None
        else:
            self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)


class SocialRecommender(GeneralRecommender):
    """This is an abstract social recommender. All the social graph model should implement this class.
    The base social recommender class provide the basic social graph dataset and parameters information.
    """
    type = ModelType.SOCIAL

    def __init__(self, config, dataset):
        super(SocialRecommender, self).__init__(config, dataset)
