import sys
sys.path.append('../')

import pandas as pd
from FCTree.FCTree import *
import os, random, pickle
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def process_cat(data, categorical_features):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    return data


def run_fctree(task, train_x, val_x, test_x, train_y, val_y, cat_features, n_new_features, n_jobs):
    n_train = train_x.shape[0]
    train_x = pd.concat([train_x, val_x], axis=0)
    train_c = train_x[cat_features]
    test_c = test_x[cat_features]
    for f in cat_features:
        del train_x[f]
        del test_x[f]
    train_y = pd.concat([train_y, val_y], axis=0)
    train_y.columns = ['label']
    if task == 'classification':
        fct = FCTree(df=train_x, label=train_y, min_leaf=10, n_e=None, is_regression=False)
    else:
        fct = FCTree(df=train_x.fillna(0), label=train_y, min_leaf=10, n_e=None, is_regression=True)
    train_x = train_x.copy()
    test_x = test_x.copy()
    for i, op in enumerate(fct.E[:n_new_features]):
        train_x.loc[:, 'new_feature_%d' % i] = op.forward(train_x).copy()
        test_x.loc[:, 'new_feature_%d' % i] = op.forward(test_x).copy()
    train_x = pd.concat([train_x, train_c], axis=1)
    test_x = pd.concat([test_x, test_c], axis=1)
    val_x = train_x[n_train:]
    train_x = train_x[:n_train]
    return train_x, val_x, test_x


if __name__ == '__main__':
    data = pd.read_csv('./train.csv', nrows=12800, usecols=['target', 'v1', 'v2', 'v3', 'v4', 'v110'])
    categorical_features = list(data.select_dtypes(exclude=np.number).columns)
    numerical_features = []
    for feature in data.columns:
        if (feature == 'target') or (feature in categorical_features):
            continue
        else:
            numerical_features.append(feature)

    print(categorical_features)
    data = process_cat(data, categorical_features)
    train_x = data[:6400]
    test_x = data[6400:]
    train_index = train_x[:int(len(train_x) * 0.75)].index
    val_index = train_x[int(len(train_x) * 0.75):].index
    train_y = train_x[['target']]
    test_y = test_x[['target']]
    del train_x['target']
    del test_x['target']
    train_x, val_x, test_x = run_fctree('regression', train_x.loc[train_index], train_x.loc[val_index], test_x,
                                      train_y.loc[train_index], train_y.loc[val_index], test_y,
                                      categorical_features, n_new_features=10)