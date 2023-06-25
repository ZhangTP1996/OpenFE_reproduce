from autofeat import AutoFeatRegressor, AutoFeatClassifier
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_autofeat(task, train_x, val_x, test_x, train_y, val_y, cat_features, n_new_features, n_jobs):
    n_train, n_feat = train_x.shape
    data = pd.concat([train_x, val_x], axis=0)
    label = pd.concat([train_y, val_y], axis=0)
    num_features = [x for x in data.columns.values.tolist() if x not in cat_features]
    print(num_features)
    print(cat_features)
    data[num_features] = data[num_features].fillna(0)
    test_x[num_features] = test_x[num_features].fillna(0)
    print(data.columns.values.tolist())
    if task == 'regression':
        afreg = AutoFeatRegressor(verbose=1, feateng_steps=1, categorical_cols=cat_features, featsel_runs=10, n_jobs=n_jobs)
        data = afreg.fit_transform(data, label)
        test_x = afreg.transform(test_x)
    elif task == 'classification':
        afreg = AutoFeatClassifier(verbose=1, feateng_steps=1, categorical_cols=cat_features, featsel_runs=10, n_jobs=n_jobs)
        data = afreg.fit_transform(data, label)
        test_x = afreg.transform(test_x)
    else:
        raise NotImplementedError("Cannot recognize task %s!" % task)
    data = data[data.columns[:n_feat+n_new_features]]
    test_x = test_x[test_x.columns[:n_feat+n_new_features]]
    train_x = data[:n_train]
    val_x = data[n_train:]
    return train_x, val_x, test_x
