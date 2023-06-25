import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from nn_utils import load_data, save_to_rtdl_format
from OpenFE import OpenFE, get_candidate_features
from utils import node_to_formula, formula_to_node, calculate_new_features
import warnings
import argparse
from copy import deepcopy
from multiprocessing import cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--fold', type=int, default=32)
parser.add_argument('--task_type', type=str, choices=['classification', 'regression'])
parser.add_argument('--n_saved_features', type=int, default=None)
parser.add_argument('--ordinal_threshold', type=int, default=100)
parser.add_argument('--remain_for_stage2', type=int, default=None)
parser.add_argument('--remain', type=int, default=2000)
parser.add_argument('--n_jobs', type=int, default=cpu_count())
parser.add_argument('--is_load', action='store_true')
warnings.filterwarnings("ignore")

args = parser.parse_args()

ALGORITHM = 'OpenFE'
TASK = args.task_type
file = args.data
n_jobs = args.n_jobs

def process_cat(X_train, X_test, cat_features):
    n_train = len(X_train)
    data = pd.concat([X_train, X_test], axis=0)
    for feature in cat_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    X_train, X_test = data[:n_train], data[n_train:]
    return X_train, X_test


if __name__ == '__main__':
    print(args)
    path = f'../data/{file}/'
    train_x, val_x, X_test, train_c, val_c, test_c, train_y, val_y, y_test = load_data(path)
    n_train = len(train_x)
    n_val = len(val_x)
    N_train = pd.concat([train_x, val_x], axis=0)
    C_train = pd.concat([train_c, val_c], axis=0)
    X_train = pd.concat([N_train, C_train], axis=1)
    X_test = pd.concat([X_test, test_c], axis=1)
    y_train = pd.concat([train_y, val_y], axis=0)

    X_train.index = range(len(X_train))
    y_train.index = range(len(y_train))
    train_index = X_train[:n_train].index
    val_index = X_train[n_train:].index

    if TASK == 'regression':
        mean = y_train[:n_train].mean()
        std = y_train[:n_train].std()
        y_train = (y_train - mean) / std

    if args.is_load is False:
        cat_features = train_c.columns.to_list()
        ord_features = []
        num_features = []
        for feature in N_train.columns:
            if N_train[feature].nunique() <= args.ordinal_threshold:
                ord_features.append(feature)
            else:
                num_features.append(feature)
        print("ordinal features")
        print(ord_features)
        candidate_features_list = get_candidate_features(numerical_features=num_features,
                                                         categorical_features=cat_features,
                                                         ordinal_features=ord_features)

        if TASK == 'classification':
            if y_train[y_train.columns[0]].nunique() > 2:
                metric = 'multi_logloss'
                # metric = 'multi_error'
            else:
                metric = 'binary_logloss'
        else:
            metric = 'rmse'
        np.save('./all_candidate_features.npy', np.array([node_to_formula(node) for node in candidate_features_list]))
        ofe = OpenFE()
        features = ofe.fit(data=X_train, label=y_train,
                           candidate_features_list=candidate_features_list,
                           metric=metric,
                           train_index=train_index, val_index=val_index,
                           categorical_features=cat_features,
                           remain_for_stage2=args.remain_for_stage2,
                           remain=args.remain,
                           n_jobs=n_jobs, fold=args.fold, task=TASK)
    else:
        ofe = OpenFE()
        ofe.n_jobs = args.n_jobs
        features = np.load('./all_features.npy')
        features = [[formula_to_node(f), score] for f, score in features]
        ofe.new_features_list = features
    if TASK == 'regression':
        y_train = y_train * std + mean
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    y_train_copy = y_train.copy()
    if args.n_saved_features is not None:
        # X_train, X_test = ofe.transform(X_train_copy, X_test_copy, n_new_features=args.n_saved_features)
        new_features = [feature for feature, _ in ofe.new_features_list[:args.n_saved_features]]
        X_train, X_test = calculate_new_features(X_train_copy, X_test_copy,
                                                 new_features, n_jobs=args.n_jobs)
        print(X_train.shape, n_train)
        print(X_train)
        cat_features = list(X_train.select_dtypes(exclude=np.number).columns)
        num_features = list(X_train.select_dtypes(include=np.number).columns)
        X_train, X_val = X_train[:n_train], X_train[n_train:]
        y_train, y_val = y_train_copy[:n_train], y_train_copy[n_train:]
        N_train, c_train = X_train[num_features], X_train[cat_features]
        N_val, c_val = X_val[num_features], X_val[cat_features]
        N_test, c_test = X_test[num_features], X_test[cat_features]

        file_path = f'../data/{file}-{ALGORITHM}-{args.n_saved_features}/'
        if TASK == 'classification':
            if y_train[y_train.columns[0]].nunique() <= 2:
                task_type = 'binclass'
            else:
                task_type = 'multiclass'
        else:
            task_type = 'regression'
        save_to_rtdl_format(file_name=file,
                            file_path=file_path,
                            task_type=task_type,
                            train_x=N_train, val_x=N_val, test_x=N_test,
                            train_c=c_train, val_c=c_val, test_c=c_test,
                            train_y=y_train, val_y=y_val, test_y=y_test)

