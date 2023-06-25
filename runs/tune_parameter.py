import os
from pathlib import Path
import json
import logging
import argparse
from datetime import datetime
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--model', type=str, default='lightgbm')
parser.add_argument('--model_type', type=str, default='default')
parser.add_argument('--task_type', type=str, choices=['classification', 'regression'])
parser.add_argument('--n_saved_features', type=int, default=None)

args = parser.parse_args()
ALGORITHM = 'OpenFE'
TASK = args.task_type
file = args.data
model = args.model
model_type = args.model_type  # tuned

if args.n_saved_features is None:
    new_file = f'{file}'
else:
    new_file = f'{file}-{ALGORITHM}-{args.n_saved_features}'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=f'{args.data}.log',
                        format="%(levelname)s:%(asctime)s:%(message)s")
    logging.info('>>>> start trail for {}'.format(new_file))
    for model in [model]:
        os.makedirs(f'output/{new_file}/{model}/tuning/0', exist_ok=True)
        os.makedirs(f'output/{new_file}/{model}/tuned', exist_ok=True)
        # grid search according to original paper: https://arxiv.org/pdf/1909.06312.pdf
        if model == 'node':
            param_grid = {'num_layers': [2, 4, 8], 'tree_count': [1024, 2048], 'tree_depth': [6, 8], 'output_dim': [2, 3]}
            node_tuning_results = {}
            start = datetime.now()
            for ctn in param_grid['tree_count']:
                for layer in param_grid['num_layers']:
                    for depth in param_grid['tree_depth']:
                        for dim in param_grid['output_dim']:
                            os.system(f"CUDA_VISIBLE_DEVICES=2,3 python bin/{model}.py --force \
                                output/{new_file}/{model}/tuning/0/t{ctn}_l{layer}_de{depth}_di{dim}.toml")
                            result = json.load(open(f'output/{new_file}/{model}/tuning/0/t{ctn}_l{layer}_de{depth}_di{dim}/stats.json'))
                            node_tuning_results[f"t{ctn}_l{layer}_de{depth}_di{dim}"] = result['metrics']['test']['score']
                            logging.info(f"t{ctn}_l{layer}_de{depth}_di{dim}: {result['metrics']['test']['score']}")
        else:
            p = Path(f'output/{new_file}/{model}/tuning/0.toml')
            p.write_text(p.read_text().replace(f"path = 'data/{file}'",
                                            f"path = 'data/{new_file}'"))
            start = datetime.now()
            os.system(f'CUDA_VISIBLE_DEVICES=0,1,2 python bin/tune_copy.py --force output/{new_file}/{model}/tuning/0.toml')

        logging.info(f"Time for tuning {new_file} {model}: %s" % (datetime.now() - start))

        if model == 'node':
            best_config_name = max(node_tuning_results, key=lambda x: node_tuning_results[x])
            logging.info(f"The best config for {model} is: {best_config_name}")
            open(f'output/{new_file}/{model}/tuned/0.toml', 'w').write(
                open(f'output/{new_file}/{model}/tuning/0/{best_config_name}.toml').read()
            )
        else:
            open(f'output/{new_file}/{model}/tuned/0.toml', 'w').write(
                open(f'output/{new_file}/{model}/tuning/0/best.toml').read()
            )

        if model == 'xgboost' or model == 'lightgbm':
            res = os.system(f'python bin/{model}_.py output/{new_file}/{model}/tuned/0.toml')
        else:
            res = os.system(
                f'CUDA_VISIBLE_DEVICES=0,1,2 python bin/{model}.py output/{new_file}/{model}/tuned/0.toml')
        result = json.load(open(f'output/{new_file}/{model}/tuned/0/stats.json'))
        try:
            logging.info(f"The result for {model} is")
            if 'class' in TASK:
                logging.info('acc: ' + str(result['metrics']['test']['accuracy']))
            else:
                logging.info('rmse: ' + str(result['metrics']['test']['rmse']))
        except:
            raise ValueError(TASK)
        logging.info('end trail <<<<')
