import pandas as pd
from OpenFE import automatic_feature_generation, get_candidate_features, get_residual_label
import numpy as np
from utils import calculate_new_features
import gc
import xgboost as xgb
import warnings
from autofeat import AutoFeatRegressor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
warnings.filterwarnings("ignore")


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
    # afreg = AutoFeatRegressor(verbose=1, feateng_steps=1, categorical_cols=categorical_features, featsel_runs=10)
    # # train on noisy data
    # print(train_x.csv.shape)
    # df = afreg.fit_transform(train_x.csv.fillna(0), train_y)
    # print(df.shape)
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
    auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
