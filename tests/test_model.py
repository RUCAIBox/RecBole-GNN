import os
import unittest

from recbole_gnn.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]


def quick_test(config_dict):
    objective_function(config_dict=config_dict, config_file_list=config_file_list, saved=False)


class TestGeneralRecommender(unittest.TestCase):
    def test_bpr(self):
        config_dict = {
            'model': 'BPR',
        }
        quick_test(config_dict)

    def test_neumf(self):
        config_dict = {
            'model': 'NeuMF',
        }
        quick_test(config_dict)

    def test_ngcf(self):
        config_dict = {
            'model': 'NGCF',
        }
        quick_test(config_dict)

    def test_lightgcn(self):
        config_dict = {
            'model': 'LightGCN',
        }
        quick_test(config_dict)

    def test_sgl(self):
        config_dict = {
            'model': 'SGL',
        }
        quick_test(config_dict)

    def test_hmlet(self):
        config_dict = {
            'model': 'HMLET',
        }
        quick_test(config_dict)

    def test_ncl(self):
        config_dict = {
            'model': 'NCL',
            'num_clusters': 10
        }
        quick_test(config_dict)

    def test_simgcl(self):
        config_dict = {
            'model': 'SimGCL'
        }
        quick_test(config_dict)

    def test_xsimgcl(self):
        config_dict = {
            'model': 'XSimGCL'
        }
        quick_test(config_dict)

    def test_lightgcl(self):
        config_dict = {
            'model': 'LightGCL'
        }
        quick_test(config_dict)

    def test_directau(self):
        config_dict = {
            'model': 'DirectAU'
        }
        quick_test(config_dict)

    def test_ssl4rec(self):
        config_dict = {
            'model': 'SSL4REC'
        }
        quick_test(config_dict)


class TestSequentialRecommender(unittest.TestCase):
    def test_gru4rec(self):
        config_dict = {
            'model': 'GRU4Rec',
        }
        quick_test(config_dict)

    def test_narm(self):
        config_dict = {
            'model': 'NARM',
        }
        quick_test(config_dict)

    def test_sasrec(self):
        config_dict = {
            'model': 'SASRec',
        }
        quick_test(config_dict)

    def test_srgnn(self):
        config_dict = {
            'model': 'SRGNN',
        }
        quick_test(config_dict)

    def test_srgnn_uni100(self):
        config_dict = {
            'model': 'SRGNN',
            'eval_args': {
                'split': {'LS': "valid_and_test"},
                'mode': 'uni100',
                'order': 'TO'
            }
        }
        quick_test(config_dict)

    def test_gcsan(self):
        config_dict = {
            'model': 'GCSAN',
        }
        quick_test(config_dict)

    def test_niser(self):
        config_dict = {
            'model': 'NISER',
        }
        quick_test(config_dict)

    def test_lessr(self):
        config_dict = {
            'model': 'LESSR'
        }
        quick_test(config_dict)

    def test_tagnn(self):
        config_dict = {
            'model': 'TAGNN'
        }
        quick_test(config_dict)

    def test_gcegnn(self):
        config_dict = {
            'model': 'GCEGNN'
        }
        quick_test(config_dict)

    def test_sgnnhn(self):
        config_dict = {
            'model': 'SGNNHN'
        }
        quick_test(config_dict)


class TestSocialRecommender(unittest.TestCase):
    def test_diffnet(self):
        config_dict = {
            'model': 'DiffNet',
        }
        quick_test(config_dict)

    def test_mhcn(self):
        config_dict = {
            'model': 'MHCN',
        }
        quick_test(config_dict)

    def test_sept(self):
        config_dict = {
            'model': 'SEPT',
        }
        quick_test(config_dict)


if __name__ == '__main__':
    unittest.main()
