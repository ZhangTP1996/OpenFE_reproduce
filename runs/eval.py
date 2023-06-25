import os
from pathlib import Path
import json
import logging
import argparse
import numpy as np
from multiprocessing import cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--model', type=str, default='lightgbm')
parser.add_argument('--model_type', type=str, default='default')
parser.add_argument('--task_type', type=str, choices=['classification', 'regression'])
parser.add_argument('--algorithm', type=str, default=None)
parser.add_argument('--n_first_order', type=int, default=None)
parser.add_argument('--n_saved_features', type=int, default=None)

args = parser.parse_args()
ALGORITHM = args.algorithm
TASK = args.task_type
file = args.data
model = args.model
model_type = args.model_type


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=f'{args.algorithm}-{args.data}.log',
                        format="%(levelname)s:%(asctime)s:%(message)s")
    logging.info('========== start trail for {} ==========='.format(file))

    if args.algorithm is None:
        new_file = f'{file}'
    elif args.algorithm == 'OpenFE':
        if args.n_first_order is None:
            new_file = f'{file}-{ALGORITHM}-{args.n_saved_features}'
        else:
            new_file = f'{file}-{ALGORITHM}-{args.n_first_order}-{args.n_saved_features}'
    else:
        new_file = f'{file}-{ALGORITHM}-{args.n_saved_features}'

    os.makedirs(f'tuned_parameters/{new_file}/{model}/{model_type}/', exist_ok=True)
    metric_list = []
    for seed in range(10):
        try:
            os.system(f'cp tuned_parameters/{file}/{model}/{model_type}/{seed}.toml'
                      f' tuned_parameters/{new_file}/{model}/{model_type}/{seed}.toml')
            p = Path(f'tuned_parameters/{new_file}/{model}/{model_type}/{seed}.toml')
            p.write_text(p.read_text().replace(f"path = 'data/{file}'",
                                               f"path = 'data/{new_file}'"))
            os.makedirs(f'tuned_parameters/{new_file}/{model}/{model_type}/{seed}/', exist_ok=True)
            if model in ['xgboost', 'lightgbm']:
                res = os.system(
                    f'CUDA_VISIBLE_DEVICES=2,3 python3 bin/{model}_.py --force tuned_parameters/{new_file}/{model}/{model_type}/{seed}.toml')
            else:
                res = os.system(
                    f'CUDA_VISIBLE_DEVICES=1,2,3 python3 bin/{model}.py --force tuned_parameters/{new_file}/{model}/{model_type}/{seed}.toml')
            result = json.load(open(f'tuned_parameters/{new_file}/{model}/{model_type}/{seed}/stats.json'))
            os.system(f'rm tuned_parameters/{new_file}/{model}/{model_type}/{seed}/*.npy')
            os.system(f'rm tuned_parameters/{new_file}/{model}/{model_type}/{seed}/*.pickle')
            os.makedirs(f'output/{new_file}/{model}/{model_type}/{seed}', exist_ok=True)
            os.system(f'cp tuned_parameters/{new_file}/{model}/{model_type}/{seed}/*.json output/{new_file}/{model}/{model_type}/{seed}/')
            logging.info(args)
            if 'class' in TASK:
                open(f'output/{new_file}/{model}/{model_type}/{seed}/result','w').write('accuracy: ' + str(result['metrics']['test']['accuracy']) + '\n')
                if 'roc_auc' in result['metrics']['test']:
                    open(f'output/{new_file}/{model}/{model_type}/{seed}/result','a').write('roc_auc: ' + str(result['metrics']['test']['roc_auc']))
                    logging.info(f'AUC of seed {seed}: ' + str(result['metrics']['test']['roc_auc']))
                    metric_list.append(result['metrics']['test']['roc_auc'])
                else:
                    logging.info(f'accuracy of seed {seed}: ' + str(result['metrics']['test']['accuracy']))
                    metric_list.append(result['metrics']['test']['accuracy'])
            else:
                open(f'output/{new_file}/{model}/{model_type}/{seed}/result','w').write('rmse: ' + str(result['metrics']['test']['rmse']))
                logging.info(f'rmse of seed {seed}: ' + str(result['metrics']['test']['rmse']))
                metric_list.append(result['metrics']['test']['rmse'])
        except:
            import traceback
            logging.info(traceback.format_exc())
    logging.info(f'The average metric value is {np.mean(metric_list)} with std {np.std(metric_list)}')
