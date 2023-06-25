import sys
sys.path.append('../')
from SAFE.safe import *
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_safe(task, train_x, val_x, test_x, train_y, val_y, cat_features, n_new_features, n_jobs):
    SS = MinMaxScaler()
    n_train = train_x.shape[0]
    train_x = pd.concat([train_x, val_x], axis=0)
    train_x.reset_index(drop=True, inplace=True)
    train_c = train_x[cat_features]
    test_c = test_x[cat_features]
    for f in cat_features:
        del train_x[f]
        del test_x[f]
    train_y = pd.concat([train_y, val_y], axis=0)
    train_y.columns = ['label']
    train_y.reset_index(drop=True, inplace=True)
    train_x = SS.fit_transform(train_x)
    test_x = SS.transform(test_x)

    set_globals(train_x, test_x, train_y)

    if task == 'classification':
        gbm = xgb.XGBClassifier(**{'n_jobs': n_jobs, 'random_state': 1, 'n_estimators': 10,
                                   'importance_type': 'gain', 'use_label_encoder': False})
    else:
        gbm = xgb.XGBRegressor(**{'n_jobs': n_jobs, 'random_state': 1, 'n_estimators': 10,
                                   'importance_type': 'gain', 'use_label_encoder': False})
    # gbm.fit(train_x.csv, train_y, eval_set=[(val_x, val_y)])
    gbm.fit(train_x, train_y.values.ravel())

    combinations_from_path = get_feature_combinations(gbm)
    train_x_new = calculate_according_to_combinations(combinations_from_path, n_jobs)
    if train_x_new is not None:
        train_x_new = delete_correlated_features(train_x_new)

        if task == 'classification':
            gbm = xgb.XGBClassifier(**{'n_jobs': n_jobs, 'random_state': 1, 'n_estimators': 100,
                                       'importance_type': 'gain', 'use_label_encoder': False})
        else:
            gbm = xgb.XGBRegressor(**{'n_jobs': n_jobs, 'random_state': 1, 'n_estimators': 100,
                                      'importance_type': 'gain', 'use_label_encoder': False})
        gbm.fit(train_x_new, train_y.values.ravel())
        new_features = []
        for feature, imp in zip(train_x_new.columns, gbm.feature_importances_):
            new_features.append([feature, imp])
        new_features = sorted(new_features, key=lambda x: x[1], reverse=True)
        new_features = [item[0] for item in new_features[:n_new_features]]
        print("The number of new features", len(new_features))

        from SAFE.utils import get_results_by_new_features

        train_x, __, test_x = get_results_by_new_features(task, new_features, train_x, test_x, test_x,
                                                              train_y, n_jobs=n_jobs, half_way=True)
    train_x = pd.concat([train_x, train_c], axis=1)
    test_x = pd.concat([test_x, test_c], axis=1)
    val_x = train_x[n_train:]
    train_x = train_x[:n_train]
    return train_x, val_x, test_x


def process_cat(data, categorical_features):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    return data


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
    train_x, val_x, test_x = run_safe('regression', train_x.loc[train_index], train_x.loc[val_index], test_x,
                                      train_y.loc[train_index], train_y.loc[val_index], test_y,
                                      categorical_features, n_new_features=10)