import pandas as pd
import numpy as np
import sys, os
sys.path.append('../')
from OpenFE import OpenFE, get_candidate_features
from utils import node_to_formula, calculate_new_features
import warnings
import argparse
from copy import deepcopy
from multiprocessing import cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=cpu_count())
warnings.filterwarnings("ignore")

args = parser.parse_args()

def process_cat(X_train, X_test, cat_features):
    n_train = len(X_train)
    data = pd.concat([X_train, X_test], axis=0)
    for feature in cat_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    X_train, X_test = data[:n_train], data[n_train:]
    return X_train, X_test

def process_cat(data, categorical_features):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    return data

import pickle, random

np.random.seed(23333)
random.seed(1000000007)

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename
 
def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def prepare_candidates(candidate_features_list):
    random.shuffle(candidate_features_list)
    n_comb = 0
    for i, cand in enumerate(candidate_features_list):
        if 'Combine(' in node_to_formula(cand):
            candidate_features_list[n_comb], candidate_features_list[i] = candidate_features_list[i], candidate_features_list[n_comb]
            n_comb += 1
    candidate_features_list = candidate_features_list[:2000]
    return candidate_features_list

train_x, train_y, test_x, test_y = load_variable('./cache/data.pkl')

categorical_features = list(train_x.select_dtypes(exclude=np.number).columns)
numerical_features = list(train_x.select_dtypes(include=np.number).columns)
ordinal_features = []

ofe = OpenFE()
stage2_params = {"n_estimators": 2000, "importance_type": "gain", "num_leaves": 256,
        "seed": 1, "n_jobs": args.n_jobs, "min_data_per_group": 50, "max_cat_threshold": 1024}

candidate_features_list = get_candidate_features(numerical_features=numerical_features,
                                                categorical_features=categorical_features,
                                                ordinal_features=ordinal_features)
candidate_features_list = prepare_candidates(candidate_features_list)

metric = 'binary_logloss'
train_index = []
val_index = []
for i in train_x.index:
    if np.random.random() < 0.75:
        train_index.append(i)
    else:
        val_index.append(i)

features = ofe.fit(data=train_x, label=train_y,
                    candidate_features_list=candidate_features_list,
                    metric='binary_logloss',
                    train_index=train_index, val_index=val_index,
                    categorical_features=categorical_features,
                    remain_for_stage2=None,
                    remain=2000,
                    n_jobs=args.n_jobs, fold=8, task='classification',
                    stage2_params = stage2_params)

save_variable(ofe.new_features_list, './cache/new_feature_list_order_2.pkl')
for feature, _ in ofe.new_features_list:
    print(node_to_formula(feature), _)

X_train_copy = train_x.copy()
X_test_copy = test_x.copy()
y_train_copy = train_y.copy()

new_features = [feature for feature, _ in ofe.new_features_list[:100] if _ > 0]
X_train, X_test = calculate_new_features(X_train_copy, X_test_copy,
                                            new_features, n_jobs=args.n_jobs, name='_order2')

for c in X_train.columns:
    print(c, X_train[c].dtype)

save_variable((X_train, train_y, X_test, test_y), './cache/data_order_2.pkl')
