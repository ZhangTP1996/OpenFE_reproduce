import sys, os
sys.path.append('../')
import pandas as pd
import numpy as np

from utils import node_to_formula, formula_to_node, stupid_file_to_node, calculate_new_features
import pickle
import warnings
warnings.filterwarnings("ignore")

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

def process_cat(data, categorical_features):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].astype('str')
        data[feature] = data[feature].astype('category')
    return data

X_train, train_y, X_test, test_y = load_variable('./cache/data.pkl')

cat_feats = [c for c in X_train.columns if str(X_train[c].dtype)=='category']
num_feats = [c for c in X_train.columns if str(X_train[c].dtype)!='category']

X_train = X_train[cat_feats+num_feats]
X_test = X_test[cat_feats+num_feats]


if True:
    X_train_2, train_y_2, X_test_2, test_y_2 = load_variable('./cache/data_order_2.pkl')
    for c in [cc for cc in X_train_2.columns if ('_order2' in cc)][:2000]:
        if str(X_train_2[c].dtype)=='category':
            cat_feats.append(c)
            X_train[c] = X_train_2[c]
            X_test[c] = X_test_2[c]

    for c in [cc for cc in X_train_2.columns if ('_order2' in cc)][:1000]:
        if str(X_train_2[c].dtype)!='category' and len(cat_feats)+len(num_feats)<325:
            num_feats.append(c)
            X_train[c] = X_train_2[c]
            X_test[c] = X_test_2[c]

X_train = process_cat(X_train, cat_feats)
X_test = process_cat(X_test, cat_feats)

print('length', len(num_feats), len(cat_feats))

import catboost as cat
import numpy as np

def doit(new_train, Y, cat_feats, seed=432013):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "learning_rate": 0.03,
        "iterations": 3000,
        "l2_leaf_reg": 3,
        "random_seed": seed,
        "subsample": 0.66,
        "od_type": "Iter",
        "rsm": 0.2,
        "depth": 6,
        "border_count": 128,
    }
    model = cat.CatBoostClassifier(**params)
    train_data = cat.Pool(new_train.iloc[: Y.shape[0]], label=Y, cat_features=cat_feats)
    fit_model = model.fit(train_data, verbose=100)
    return fit_model

for ep in range(1):
    fit_model = doit(X_train, train_y.values.ravel(), cat_feats, ep*233+666)
    y_pred = fit_model.predict_proba(X_test)
    hyperpath = '../data/BNP'
    submission = pd.read_csv(f"{hyperpath}/sample_submission.csv.zip")
    submission.loc[:, "PredictedProb"] = y_pred[:, 1]
    os.makedirs('./result', exist_ok=True)
    submission.to_csv(f"./result/output_high_order_{ep}.csv", index=False)
