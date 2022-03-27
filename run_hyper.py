import argparse

from recbole.trainer import HyperTuning
from recbole_gnn.quick_start import objective_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default=None, help='fixed config files')
    parser.add_argument('--params_file', type=str, default=None, help='parameters file')
    parser.add_argument('--output_file', type=str, default='hyper_example.result', help='output file')
    args, _ = parser.parse_known_args()

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hp = HyperTuning(objective_function, algo='exhaustive',
                     params_file=args.params_file, fixed_config_file_list=config_file_list)
    hp.run()
    hp.export_result(output_file=args.output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    main()
