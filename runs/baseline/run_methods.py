from autocross.run_autocross import run_autocross
from myautofeat.run_autofeat import run_autofeat
from FCTree.run_fctree import run_fctree
from SAFE.run_safe import run_safe
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
from nn_utils import save_to_rtdl_format, load_data
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True, choices=['autocross', 'fctree', 'safe', 'autofeat'])
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--task', type=str, choices=['classification', 'regression'])
parser.add_argument('--n_new_features', type=int, required=True)
parser.add_argument('--n_jobs', type=int, default=20)

args = parser.parse_args()

def prepare_data(args):
    path = '../../data/%s/' % args.data
    train_x, val_x, test_x, train_c, val_c, test_c, train_y, val_y, test_y = load_data(path)
    if train_x.isnull().any().sum() != 0: 
        train_x.fillna(value=0, inplace=True)
    if val_x.isnull().any().sum() != 0 :
        val_x.fillna(value=0, inplace=True)
    if test_x.isnull().any().sum() != 0:
        test_x.fillna(value=0, inplace=True)

    train_c = train_c.astype('category')
    val_c = val_c.astype('category')
    test_c = test_c.astype('category')
    cat_features = list(train_c.columns)
    train_x = pd.concat([train_x, train_c], axis=1)
    val_x = pd.concat([val_x, val_c], axis=1)
    test_x = pd.concat([test_x, test_c], axis=1)
    return train_x, val_x, test_x, train_y, val_y, test_y, cat_features


def save_data(args, train_x:pd.DataFrame, val_x, test_x, train_y, val_y, test_y, save_path):
    cat_features = list(train_x.select_dtypes(exclude=np.number).columns)
    num_features = list(train_x.select_dtypes(include=np.number).columns)
    train_c = train_x[cat_features]
    val_c = val_x[cat_features]
    test_c = test_x[cat_features]
    train_x = train_x[num_features]
    val_x = val_x[num_features]
    test_x = test_x[num_features]
    if args.task == 'classification':
        if train_y[train_y.columns[0]].nunique() <= 2:
            task_type = 'binclass'
        else:
            task_type = 'multiclass'
    else:
        task_type = 'regression'
    save_to_rtdl_format(args.data, save_path, task_type, train_x, val_x, test_x,
                        train_c, val_c, test_c, train_y, val_y, test_y)


if __name__ == '__main__':
    print("="*8 + args.data + ': ' + args.task)
    train_x, val_x, test_x, train_y, val_y, test_y, cat_features = prepare_data(args)
    save_path = '../../data/%s-%s-%s/' % (args.data, args.method, args.n_new_features)
    os.makedirs(save_path, exist_ok=True)
    info = {}
    info['args'] = vars(args) #args
    start = datetime.now()
    if args.method == 'autocross':
        train_x, val_x, test_x = run_autocross(train_x, val_x, test_x, train_y, val_y, args.n_jobs, args.n_new_features)
    elif args.method == 'fctree':
        train_x, val_x, test_x = run_fctree(args.task, train_x, val_x, test_x, train_y, val_y, cat_features, args.n_new_features, args.n_jobs)
    elif args.method == 'safe':
        train_x, val_x, test_x = run_safe(args.task, train_x, val_x, test_x, train_y, val_y, cat_features, args.n_new_features, args.n_jobs)
    elif args.method == 'autofeat':
        train_x, val_x, test_x = run_autofeat(args.task, train_x, val_x, test_x, train_y, val_y, cat_features, args.n_new_features, args.n_jobs)
    info['time'] = str(datetime.now() - start)
    print(train_x.shape)
    print(val_x.shape)
    print(test_x.shape)
    print(train_x.columns.tolist())
    print(info)
    json_str = json.dumps(info, indent=4)
    with open(os.path.join(save_path, 'stats.json'), 'w') as json_file:
        json_file.write(json_str)
    save_data(args, train_x, val_x, test_x, train_y, val_y, test_y, save_path)

