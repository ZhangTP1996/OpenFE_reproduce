import os
from pathlib import Path
import json
import logging
import argparse
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
    logging.basicConfig(level=logging.INFO, filename=f'{args.data}.log',
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

    try:
        os.system(f'cp tuned_parameters/{file}/{model}/{model_type}/0.toml'
                  f' tuned_parameters/{new_file}/{model}/{model_type}/0.toml')
        p = Path(f'tuned_parameters/{new_file}/{model}/{model_type}/0.toml')
        p.write_text(p.read_text().replace(f"path = 'data/{file}'",
                                           f"path = 'data/{new_file}'"))
        os.makedirs(f'tuned_parameters/{new_file}/{model}/{model_type}/0/', exist_ok=True)
        if model in ['xgboost', 'lightgbm']:
            res = os.system(
                f'CUDA_VISIBLE_DEVICES=2,3 python3 bin/{model}_.py --force tuned_parameters/{new_file}/{model}/{model_type}/0.toml')
        else:
            res = os.system(
                f'CUDA_VISIBLE_DEVICES=1,2,3 python3 bin/{model}.py --force tuned_parameters/{new_file}/{model}/{model_type}/0.toml')
        result = json.load(open(f'tuned_parameters/{new_file}/{model}/{model_type}/0/stats.json'))
        os.system(f'rm tuned_parameters/{new_file}/{model}/{model_type}/0/*.npy')
        os.system(f'rm tuned_parameters/{new_file}/{model}/{model_type}/0/*.pickle')
        os.makedirs(f'output/{new_file}/{model}/{model_type}/', exist_ok=True)
        os.system(f'cp tuned_parameters/{new_file}/{model}/{model_type}/0/*.json output/{new_file}/{model}/{model_type}/')
        logging.info(args)
        if 'class' in TASK:
            open(f'output/{new_file}/{model}/{model_type}/result','w').write('accuracy: ' + str(result['metrics']['test']['accuracy']) + '\n')
            if 'roc_auc' in result['metrics']['test']:
                open(f'output/{new_file}/{model}/{model_type}/result','a').write('roc_auc: ' + str(result['metrics']['test']['roc_auc']))
                logging.info('AUC: ' + str(result['metrics']['test']['roc_auc']))
            logging.info('accuracy: ' + str(result['metrics']['test']['accuracy']))
        else:
            open(f'output/{new_file}/{model}/{model_type}/result','w').write('rmse: ' + str(result['metrics']['test']['rmse']))
            logging.info('rmse: ' + str(result['metrics']['test']['rmse']))
    except:
        import traceback
        logging.info(traceback.format_exc())
