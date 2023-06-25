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


def read_data():
    USECOLS = [
        "v10", "v12", "v14", "v21", "v22", "v24", "v30", "v31",
        "v34", "v38", "v40", "v47", "v50", "v52", "v56", "v62",
        "v66", "v72", "v75", "v79", "v91", "v112", "v113", "v114", "v129",
    ]
    perm = np.random.permutation(len(pd.read_csv(f'{hyperpath}/train.csv.zip', usecols=USECOLS+["target"])))
    df = pd.read_csv(f'{hyperpath}/train.csv.zip', usecols=USECOLS+["target"]).iloc[perm].reset_index(drop=True)
    df_test = pd.read_csv(f'{hyperpath}/test.csv.zip', usecols=USECOLS).reset_index(drop=True)
    categorical_features = list(df.select_dtypes(exclude=np.number).columns)
    numerical_features = []
    for feature in df.columns:
        if (feature == 'target') or (feature in categorical_features): continue
        else: numerical_features.append(feature)
    ordinal_features = []
    return pd.concat([df, df_test]).reset_index(drop=True), len(df), categorical_features, numerical_features, ordinal_features


def prepare_candidates(candidate_features_list):
    random.shuffle(candidate_features_list)
    n_comb = 0
    for i, cand in enumerate(candidate_features_list):
        if 'Combine(' in node_to_formula(cand):
            candidate_features_list[n_comb], candidate_features_list[i] = candidate_features_list[i], candidate_features_list[n_comb]
            n_comb += 1
    candidate_features_list = candidate_features_list[:3000]
    return candidate_features_list

if __name__ == '__main__':
    print(args)

    hyperpath = '../data/BNP'

    data, len_T, categorical_features, numerical_features, ordinal_features = read_data()
    data = process_cat(data, categorical_features)

    train_x = data[:len_T]
    test_x = data[len_T:]
    train_index = train_x[:int(len(train_x) * 0.75)].index
    val_index = train_x[int(len(train_x) * 0.75):].index
    train_y = train_x[['target']]
    test_y = test_x[['target']]
        
    del train_x['target']
    del test_x['target']

    candidate_features_list = get_candidate_features(numerical_features=numerical_features,
                                                        categorical_features=categorical_features,
                                                        ordinal_features=ordinal_features)
    candidate_features_list = prepare_candidates(candidate_features_list)

    ofe = OpenFE()
    stage2_params = {"n_estimators": 2000, "importance_type": "gain", "num_leaves": 360,
            "seed": 1, "n_jobs": args.n_jobs, "min_data_per_group": 20, "max_cat_threshold": 4096}

    features = ofe.fit(data=train_x, label=train_y,
                        candidate_features_list=candidate_features_list,
                        metric='binary_logloss',
                        train_index=train_index, val_index=val_index,
                        categorical_features=categorical_features,
                        remain_for_stage2=None,
                        remain=2000,
                        n_jobs=args.n_jobs, fold=1, task='classification',
                        stage2_params = stage2_params)
    
    os.makedirs('./cache', exist_ok=True)
    save_variable(ofe.new_features_list, './cache/new_feature_list.pkl')
    for feature, _ in ofe.new_features_list:
        print(node_to_formula(feature), _)
    
    X_train_copy = train_x.copy()
    X_test_copy = test_x.copy()
    y_train_copy = train_y.copy()
    
    new_features = [feature for feature, _ in ofe.new_features_list[:200] if _ > 0]
    X_train, X_test = calculate_new_features(X_train_copy, X_test_copy,
                                                new_features, n_jobs=args.n_jobs)

    save_variable((X_train, train_y, X_test, test_y), './cache/data.pkl')