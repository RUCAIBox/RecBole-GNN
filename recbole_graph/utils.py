import os
import pickle
import importlib
from logging import getLogger
from recbole.data.utils import create_dataset as create_recbole_dataset
from recbole.utils import set_color
from recbole.utils import get_model as get_recbole_model
from recbole.utils.argument_list import dataset_arguments


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module('recbole_graph.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')

        default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
        file = config['dataset_save_path'] or default_file
        if os.path.exists(file):
            with open(file, 'rb') as f:
                dataset = pickle.load(f)
            dataset_args_unchanged = True
            for arg in dataset_arguments + ['seed', 'repeatable']:
                if config[arg] != dataset.config[arg]:
                    dataset_args_unchanged = False
                    break
            if dataset_args_unchanged:
                logger = getLogger()
                logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
                return dataset

        dataset = dataset_class(config)
        if config['save_dataset']:
            dataset.save()
        return dataset
    else:
        return create_recbole_dataset(config)


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_submodule = [
        'general_recommender', 'sequential_recommender', 'social_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recbole_graph.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        model_class = get_recbole_model(model_name)
    else:
        model_class = getattr(model_module, model_name)
    return model_class
