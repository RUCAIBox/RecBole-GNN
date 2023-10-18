import os
import recbole
from recbole.config.configurator import Config as RecBole_Config
from recbole.utils import ModelType as RecBoleModelType

from recbole_gnn.utils import get_model, ModelType


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
        if recbole.__version__ == "1.1.1":
            self.compatibility_settings()
        super(Config, self).__init__(model, dataset, config_file_list, config_dict)

    def compatibility_settings(self):
        import numpy as np
        np.bool = np.bool_
        np.int = np.int_
        np.float = np.float_
        np.complex = np.complex_
        np.object = np.object_
        np.str = np.str_
        np.long = np.int_
        np.unicode = np.unicode_

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
        super()._load_internal_config_dict(model, model_class, dataset)
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_init_file = os.path.join(current_path, './properties/model/' + model + '.yaml')
        quick_start_config_path = os.path.join(current_path, './properties/quick_start_config/')
        sequential_base_init = os.path.join(quick_start_config_path, 'sequential_base.yaml')
        social_base_init = os.path.join(quick_start_config_path, 'social_base.yaml')

        if os.path.isfile(model_init_file):
            config_dict = self._update_internal_config_dict(model_init_file)

        self.internal_config_dict['MODEL_TYPE'] = model_class.type
        if self.internal_config_dict['MODEL_TYPE'] == RecBoleModelType.SEQUENTIAL:
            self._update_internal_config_dict(sequential_base_init)
        if self.internal_config_dict['MODEL_TYPE'] == ModelType.SOCIAL:
            self._update_internal_config_dict(social_base_init)
