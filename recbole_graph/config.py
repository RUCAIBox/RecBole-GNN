import os

from recbole.config.configurator import Config as RecBole_Config
from recbole.utils import ModelType

from recbole_graph.utils import get_model


class Config(RecBole_Config):
    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        super(Config, self).__init__(model, dataset, config_file_list, config_dict)

    def _get_model_and_dataset(self, model, dataset):

        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset
    
    def _load_internal_config_dict(self, model, model_class, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, './properties/overall.yaml')
        model_init_file = os.path.join(current_path, './properties/model/' + model + '.yaml')
        sample_init_file = os.path.join(current_path, './properties/dataset/sample.yaml')
        dataset_init_file = os.path.join(current_path, './properties/dataset/' + dataset + '.yaml')

        quick_start_config_path = os.path.join(current_path, './properties/quick_start_config/')
        context_aware_init = os.path.join(quick_start_config_path, 'context-aware.yaml')
        context_aware_on_ml_100k_init = os.path.join(quick_start_config_path, 'context-aware_ml-100k.yaml')
        DIN_init = os.path.join(quick_start_config_path, 'sequential_DIN.yaml')
        DIN_on_ml_100k_init = os.path.join(quick_start_config_path, 'sequential_DIN_on_ml-100k.yaml')
        sequential_init = os.path.join(quick_start_config_path, 'sequential.yaml')
        special_sequential_on_ml_100k_init = os.path.join(quick_start_config_path, 'special_sequential_on_ml-100k.yaml')
        sequential_embedding_model_init = os.path.join(quick_start_config_path, 'sequential_embedding_model.yaml')
        knowledge_base_init = os.path.join(quick_start_config_path, 'knowledge_base.yaml')

        self.internal_config_dict = dict()
        for file in [overall_init_file, model_init_file, sample_init_file, dataset_init_file]:
            if os.path.isfile(file):
                config_dict = self._update_internal_config_dict(file)
                if file == dataset_init_file:
                    self.parameters['Dataset'] += [
                        key for key in config_dict.keys() if key not in self.parameters['Dataset']
                    ]

        self.internal_config_dict['MODEL_TYPE'] = model_class.type
        if self.internal_config_dict['MODEL_TYPE'] == ModelType.GENERAL:
            pass
        elif self.internal_config_dict['MODEL_TYPE'] in {ModelType.CONTEXT, ModelType.DECISIONTREE}:
            self._update_internal_config_dict(context_aware_init)
            if dataset == 'ml-100k':
                self._update_internal_config_dict(context_aware_on_ml_100k_init)
        elif self.internal_config_dict['MODEL_TYPE'] == ModelType.SEQUENTIAL:
            if model in ['DIN', 'DIEN']:
                self._update_internal_config_dict(DIN_init)
                if dataset == 'ml-100k':
                    self._update_internal_config_dict(DIN_on_ml_100k_init)
            elif model in ['GRU4RecKG', 'KSR']:
                self._update_internal_config_dict(sequential_embedding_model_init)
            else:
                self._update_internal_config_dict(sequential_init)
                if dataset == 'ml-100k' and model in ['GRU4RecF', 'SASRecF', 'FDSA', 'S3Rec']:
                    self._update_internal_config_dict(special_sequential_on_ml_100k_init)

        elif self.internal_config_dict['MODEL_TYPE'] == ModelType.KNOWLEDGE:
            self._update_internal_config_dict(knowledge_base_init)