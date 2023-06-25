import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, classification_report
import random


class MinMaxScaler():
    def __init__(self):
        pass

    def fit_transform(self, data):
        self.min = np.min(data)
        self.max = np.max(data)
        self.denominator = (self.max - self.min)
        self.denominator[self.denominator == 0] = 1
        return (data - self.min) / self.denominator + 1e-5

    def transform(self, data):
        # +1e-5 so that we can perform /
        return (data - self.min) / self.denominator + 1e-5


# def _calculate_corr(feature):
#     corr_list = []
#     for feature_temp in score_data.columns:
#         corr_list.append(abs(np.corrcoef(score_data[feature_temp].values, score_data[feature].values)[0, 1]))
#     corr_list = np.array(corr_list)
#     return corr_list

def get_score(df: pd.DataFrame, task='classification'):
    if task == 'classification':
        return df.mean()
    elif task == 'regression':
        diff_prediction = df['label']
        del df['label']
        diff_features = df.copy()
        diff_prediction_features = (diff_features.T * diff_prediction.values.ravel()).T
        return diff_prediction_features.mean() / diff_prediction.mean() - \
               (diff_features.mean() - diff_prediction_features.mean()) / (1 - diff_prediction.mean())
    else:
        print("Cannot recognize task %s." % task)
        raise NotImplementedError


def remove_redundant_features(df: pd.DataFrame, ori_features: list, top, task, threshold=0.9, n_jobs=1):
    # global score_data
    score = get_score(df, task)
    print(score)
    score = score[score > 0]
    score_list = [[feature, score[feature]] for feature in score.index]
    score_list = sorted(score_list, key=lambda x:x[1], reverse=True)
    features = [feature for feature, _ in score_list if feature not in ori_features]
    features = ori_features + features
    score_data = df[features].astype(float)

    corr_array = np.corrcoef(score_data.values.T)
    new_features = []
    idx_list = list(range(len(ori_features)))
    for idx, feature in enumerate(features[len(ori_features):]):
        idx = idx + len(ori_features)
        if len(new_features) == top:
            break
        if np.max(corr_array[idx][idx_list]) > threshold:
            continue
        else:
            new_features.append(features[idx])
            idx_list.append(idx)
    scores = [[feature, score_data[feature].mean()] for feature in new_features]
    logging.info("The number of new features is %d." % len(new_features))
    return new_features, scores


def calculate_new_features(new_feature):
    if '+' in new_feature:
        feature1, feature2 = new_feature.split('+')
        return new_feature, train_x[feature1] + train_x[feature2], \
               val_x[feature1] + val_x[feature2], \
               test_x[feature1] + test_x[feature2]
    if '-' in new_feature:
        feature1, feature2 = new_feature.split('-')
        return new_feature, train_x[feature1] - train_x[feature2], \
               val_x[feature1] - val_x[feature2], \
               test_x[feature1] - test_x[feature2]
    if '*' in new_feature:
        feature1, feature2 = new_feature.split('*')
        return new_feature, train_x[feature1] * train_x[feature2], \
               val_x[feature1] * val_x[feature2], \
               test_x[feature1] * test_x[feature2]
    if '/' in new_feature:
        feature1, feature2 = new_feature.split('/')
        return new_feature, train_x[feature1] / train_x[feature2], \
               val_x[feature1] / val_x[feature2], \
               test_x[feature1] / test_x[feature2]


def get_results_by_new_features(task, new_features, train_data, val_data, test_data, train_y, val_y=None, test_y=None, n_jobs=1, detail=False, half_way=False):
    global train_x
    global val_x
    global test_x
    train_x = train_data
    val_x = val_data
    test_x = test_data
    new_data_results = []
    ex = ProcessPoolExecutor(n_jobs)
    for new_feature in new_features:
        new_data_results.append(ex.submit(calculate_new_features, new_feature))
    ex.shutdown(wait=True)
    new_train_x_list = []
    new_val_x_list = []
    new_test_x_list = []
    for res in new_data_results:
        res = res.result()
        new_train_x_list.append(res[1])
        new_val_x_list.append(res[2])
        new_test_x_list.append(res[3])
    if len(new_train_x_list):
        new_train_x = pd.concat(new_train_x_list, axis=1)
        new_val_x = pd.concat(new_val_x_list, axis=1)
        new_test_x = pd.concat(new_test_x_list, axis=1)
        new_train_x.columns = new_features
        new_val_x.columns = new_features
        new_test_x.columns = new_features
        train_data = pd.concat([train_x, new_train_x], axis=1)
        val_data = pd.concat([val_x, new_val_x], axis=1)
        test_data = pd.concat([test_x, new_test_x], axis=1)
    else:
        train_data = train_x
        val_data = val_x
        test_data = test_x
    # gbm = lgb.LGBMClassifier(**{'importance_type': 'gain', 'deterministic': True, 'n_jobs': n_jobs, 'seed': 1,
    #                             'n_estimators': 2000, 'early_stopping_rounds': 50})
    # gbm.fit(train_data.values, train_y.values.ravel(), eval_set=[(val_data, val_y)])

    if half_way:
        return train_data, val_data, test_data

    # if task == 'classification':
    #     params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 5, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 1, 'reg_lambda': 1}
    #     # gbm = xgb.XGBClassifier(**{'importance_type': 'gain', 'n_jobs': n_jobs, 'seed': 1,
    #     #                             'n_estimators': 2000})
    #     gbm = xgb.XGBClassifier(**params)
    # else:
    #     gbm = xgb.XGBRegressor(**{'importance_type': 'gain', 'n_jobs': n_jobs, 'seed': 1,
    #                                 'n_estimators': 2000})
    # gbm.fit(train_data.values, train_y.values.ravel(), eval_set=[(val_data.values, val_y.values.ravel())],
    #         early_stopping_rounds=100, verbose=50)
    # if task == 'classification':
    #     if len(train_y['label'].unique()) == 2:
    #         score_final = roc_auc_score(test_y, gbm.predict(test_data.values))
    #     else:
    #         score_final = accuracy_score(test_y, gbm.predict(test_data.values))
    #     report = classification_report(test_y, gbm.predict(test_data.values), output_dict=True)
    #     if detail:
    #         logging.info(report)
    # else:
    #     score_final = mean_squared_error(test_y, gbm.predict(test_data.values), squared=False)
    # logging.info("The final score is %.4lf" % score_final)
    # return score_final


def run_random(ori_features, top, random_state):
    random.seed(random_state)
    np.random.seed(random_state)

    new_features_list = []
    for i in range(len(ori_features)):
        for j in range(i + 1, len(ori_features)):
            feature1 = ori_features[i]
            feature2 = ori_features[j]
            for op in ['+', '-', '*', '/']:
                new_features_list.append('%s%s%s' % (feature1, op, feature2))

    random.shuffle(new_features_list)
    new_features = new_features_list[:top]
    return new_features








