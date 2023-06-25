import xgboost as xgb
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from sklearn.metrics import mutual_info_score
from datetime import datetime
import logging

train_x = None
test_x = None
train_y = None

def set_globals(__train_x, __test_x, __train_y):
    global train_x
    global test_x
    global train_y
    train_x = __train_x
    train_x.reset_index(drop = True, inplace = True)
    test_x = __test_x
    test_x.reset_index(drop = True, inplace = True)
    train_y = __train_y
    train_y.reset_index(drop = True, inplace = True)

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


class Node():
    def __init__(self, id, feature, split_value):
        self.id = id
        self.feature = feature
        self.split_value = split_value
        self.left_child = None
        self.right_child = None

    def _print(self):
        print(' ' * self.id + str(self.feature) + ' ' + str(self.split_value))
        if self.left_child is not None:
            self.left_child._print()
            self.right_child._print()


class df_node():
    def __init__(self, data):
        self.data = data
        self.left_child = None
        self.right_child = None

    def split_by_value(self, feature, value):
        self.left_child = df_node(self.data[self.data[feature] < value])
        self.right_child = df_node(self.data[self.data[feature] >= value])


def get_child(df, tree):
    id = tree.id
    if tree.feature == 'Leaf':
        return
    id1 = id*2+1

    df1 = df[df['Node'] == id1]
    if (len(df1) != 0):
        feature1 = df1['Feature'].iloc[0]
        split_value1 = df1['Split'].iloc[0]
        tree.left_child = Node(id1, feature1, split_value1)
        get_child(df, tree.left_child)

    id2 = id*2+2
    df2 = df[df['Node'] == id2]
    if (len(df2) != 0):
        feature2 = df2['Feature'].iloc[0]
        split_value2 = df2['Split'].iloc[0]
        tree.right_child = Node(id2, feature2, split_value2)
        get_child(df, tree.right_child)


def build_tree(df):
    df1 = df[df['Node'] == 0]
    id = 0
    feature = df1['Feature'].iloc[0]
    split_value = df1['Split'].iloc[0]
    tree = Node(id, feature, split_value)
    get_child(df, tree)
    return tree


def get_feature_combinations_from_path(tree):
    if tree.left_child is None or tree.right_child is None:
        return {}, {}
    elif tree.left_child.feature == 'Leaf' and tree.right_child.feature == 'Leaf':
        # 返回这条path上的feature和对应的split， 返回目前的feature combinations
        return {tree.feature:[tree.split_value]}, {}
    else:
        features_in_path_1, combinations1 = get_feature_combinations_from_path(tree.left_child)
        features_in_path_2, combinations2 = get_feature_combinations_from_path(tree.right_child)
        # 把两条path上的combinations合起来
        for key in combinations2:
            if key in combinations1:
                for feature in combinations1[key]:
                    combinations1[key][feature].extend(combinations2[key][feature])
            else:
                combinations1[key] = combinations2[key]
        # 把两条path上出现的features合起来
        for feature in features_in_path_2:
            if feature in features_in_path_1:
                features_in_path_1[feature].extend(features_in_path_2[feature])
            else:
                features_in_path_1[feature] = features_in_path_2[feature]
        # 把当前节点的feature加进去
        if tree.feature in features_in_path_1:
            features_in_path_1[tree.feature].append(tree.split_value)
        else:
            features_in_path_1[tree.feature] = [tree.split_value]

        for feature in features_in_path_1:
            if feature == tree.feature:
                continue
            key = [tree.feature, feature]
            key = tuple(sorted(key))
            values = [tree.split_value]
            values.extend(features_in_path_1[feature])
            if key in combinations1:
                combinations1[key][tree.feature].append(tree.split_value)
            else:
                if tree.feature in features_in_path_1:
                    values1 = features_in_path_1[tree.feature]
                    values1.append(tree.split_value)
                else:
                    values1 = [tree.split_value]
                values2 = features_in_path_1[feature]
                combinations1[key] = {}
                combinations1[key][tree.feature] = values1
                combinations1[key][feature] = values2
        return features_in_path_1, combinations1


def get_feature_combinations(gbm):
    trees_dataframe = gbm.get_booster().trees_to_dataframe()
    combinations_from_path = []
    count = 0
    for tree_id in trees_dataframe['Tree'].unique():
        df = trees_dataframe[trees_dataframe['Tree'] == tree_id]
        tree = build_tree(df)
        _, combinations = get_feature_combinations_from_path(tree)
        combinations_from_path.append(combinations)
        count += len(combinations)

    return combinations_from_path


def get_entropy(label):
    if len(label) == 0:
        return 0
    value_counts = label['label'].value_counts()
    p0 = value_counts.iloc[0] / len(label)
    if p0 == 1:
        return -(p0 * np.log2(p0))
    p1 = value_counts.iloc[1] / len(label)
    return -(p0*np.log2(p0)+p1*np.log2(p1))


def get_information_value(combinations, split_values):
    feature1, feature2 = combinations
    values1 = split_values[feature1]
    values1 = sorted(values1, reverse=True)
    data = train_x[[feature1, feature2]]

    data = pd.concat([data, train_y], axis=1)

    tree = df_node(data)
    children_list = []
    while len(values1) != 0:
        tree.split_by_value(feature1, values1[0])
        tree.data = None
        lchild = tree.left_child
        rchild = tree.right_child
        children_list.append(rchild)
        values1 = values1[1:]
        if len(values1) == 0:
            children_list.append(lchild)
        tree = lchild
    values2 = split_values[feature2]
    values2 = sorted(values2, reverse=True)
    all_children_list = []
    for tree in children_list:
        values2_temp = deepcopy(values2)
        while len(values2_temp) != 0:
            tree.split_by_value(feature2, values2_temp[0])
            tree.data = None
            lchild = tree.left_child
            rchild = tree.right_child
            all_children_list.append(rchild)
            values2_temp = values2_temp[1:]
            tree = lchild
            if len(values2_temp) == 0:
                all_children_list.append(lchild)
    assert len(all_children_list) == ((len(split_values[feature1])+1) * (len(split_values[feature2])+1))
    all_entropy = get_entropy(train_y)
    length = len(train_y)
    split_entropy = 0
    for child in all_children_list:
        split_entropy += len(child.data) / length * get_entropy(child.data)
    return combinations, split_entropy - all_entropy


def calculation(op, feature1, feature2):
    if op == '+':
        new_feature = train_x[feature1].values.ravel() + train_x[feature2].values.ravel()
        mi_score = calculate_mutual_information(new_feature)
        return '%s+%s' % (feature1, feature2), new_feature, mi_score
    if op == '-':
        new_feature = train_x[feature1].values.ravel() - train_x[feature2].values.ravel()
        mi_score = calculate_mutual_information(new_feature)
        return '%s-%s' % (feature1, feature2), new_feature, mi_score
    if op == '*':
        new_feature = train_x[feature1].values.ravel() * train_x[feature2].values.ravel()
        mi_score = calculate_mutual_information(new_feature)
        return '%s*%s' % (feature1, feature2), new_feature, mi_score
    if op == '/':
        new_feature = train_x[feature1].values.ravel() / train_x[feature2].values.ravel()
        mi_score = calculate_mutual_information(new_feature)
        return '%s/%s' % (feature1, feature2), new_feature, mi_score


def calculate_mutual_information(feature_array):
    feature_array = pd.Series(feature_array)
    tab = pd.crosstab(train_y['label'], pd.qcut(feature_array, q=10, duplicates='drop'))
    mi_score = mutual_info_score(train_y['label'].values.ravel(), feature_array.values, contingency=tab)
    return mi_score


def calculate_according_to_combinations(combinations_from_path, n_jobs):
    calculated_record = {}
    ex = ProcessPoolExecutor(n_jobs)
    results = []
    for combinations_dict in combinations_from_path:
        for combinations in combinations_dict:
            results.append(ex.submit(get_information_value, combinations, combinations_dict[combinations]))
    ex.shutdown(wait=True)
    for res in results:
        combinations, information_gain = res.result()
        if combinations in calculated_record:
            calculated_record[combinations].append(information_gain)
        else:
            calculated_record[combinations] = [information_gain]
    for key in calculated_record:
        calculated_record[key] = sorted(calculated_record[key], reverse=True)
        calculated_record[key] = calculated_record[key][0]
    calculated_record = [[key, calculated_record[key]] for key in calculated_record]
    calculated_record = sorted(calculated_record, reverse=True)
    # calculated_record = calculated_record[:top*5]
    calculated_record = [item[0] for item in calculated_record]
    train_x_new = []
    new_features = []
    results = []
    ex = ProcessPoolExecutor(n_jobs)
    for feature1, feature2 in calculated_record:
        for op in ['+', '-', '*', '/']:
            results.append(ex.submit(calculation, op, feature1, feature2))
    ex.shutdown(wait=True)
    for res in results:
        name, new_feature, mi_score = res.result()
        if mi_score > 0.005:
            train_x_new.append(pd.DataFrame({name: new_feature}))
            new_features.append([name, mi_score])
    new_features = sorted(new_features, key=lambda x:x[1], reverse=True)
    new_features = [item[0] for item in new_features]
    if len(train_x_new) == 0:
        return None
    train_x_new = pd.concat(train_x_new, axis=1)
    train_x_new = train_x_new[new_features]
    return train_x_new


def delete_correlated_features(train_x_new):
    saved_features = []
    saved_index = []
    all_features = list(train_x_new.columns)
    corr_matrix = np.corrcoef(train_x_new.fillna(0).values.T)
    for i, feature in enumerate(all_features):
        flag = 1
        for idx in saved_index:
            if abs(corr_matrix[i, idx]) > 0.99:
                flag = 0
                break
        if flag:
            saved_features.append(feature)
            saved_index.append(i)
    return train_x_new[saved_features]


def run_SAFE(train_data, train_label, top, n_jobs, random_state):
    start = datetime.now()
    global train_x
    global train_y
    train_y = train_label
    SS = MinMaxScaler()
    train_x = SS.fit_transform(train_data)

    gbm = xgb.XGBClassifier(**{'n_jobs': n_jobs, 'random_state': random_state, 'n_estimators': 10,
                               'importance_type': 'gain', 'use_label_encoder': False})

    gbm.fit(train_x, train_y.values.ravel())

    combinations_from_path = get_feature_combinations(gbm)
    train_x_new = calculate_according_to_combinations(combinations_from_path, n_jobs)
    if train_x_new is None:
        return []
    train_x_new = delete_correlated_features(train_x_new)

    gbm = xgb.XGBClassifier(**{'n_jobs': n_jobs, 'random_state': 1, 'n_estimators': 100,
                               'importance_type': 'gain', 'use_label_encoder': False})
    gbm.fit(train_x_new, train_y.values.ravel())
    new_features = []
    for feature, imp in zip(train_x_new.columns, gbm.feature_importances_):
        new_features.append([feature, imp])
    new_features = sorted(new_features, key=lambda x:x[1], reverse=True)
    new_features = [item[0] for item in new_features[:top]]

    logging.info(f"SAFE time spent {datetime.now() - start}")
    return new_features


# if __name__ == '__main__':
#     n_jobs = 16
#     top = 200
#     # path = './[kaggle]ipnetwork'
#     path = './'
#     train_data = pd.read_csv(os.path.join(path, 'train_x.csv.csv'))
#     train_y = pd.read_csv(os.path.join(path, 'train_y.csv'))
#     train_y.columns = ['label']
#     train_y['label'] -= 1
#     test_data = pd.read_csv(os.path.join(path, 'test_x.csv'))
#     test_y = pd.read_csv(os.path.join(path, 'test_y.csv'))
#     test_y.columns = ['label']
#     test_y['label'] -= 1
#
#     # train_data, val_data, train_y, val_y = train_test_split(train_data, train_y, test_size=0.2, random_state=42)
#     global train_x.csv
#     # global val_x
#     global test_x
#     SS = MinMaxScaler()
#     train_x.csv = SS.fit_transform(train_data)
#     # val_x = SS.transform(val_data)
#     test_x = SS.transform(test_data)
#
#     gbm = xgb.XGBClassifier(**{'n_jobs': n_jobs, 'random_state': 1, 'n_estimators': 10,
#                                'importance_type': 'gain', 'use_label_encoder': False})
#     # gbm.fit(train_x.csv, train_y, eval_set=[(val_x, val_y)])
#     gbm.fit(train_x.csv, train_y.values.ravel())
#
#     combinations_from_path = get_feature_combinations(gbm)
#     train_x_new = calculate_according_to_combinations(combinations_from_path, n_jobs)
#     train_x_new = delete_correlated_features(train_x_new)
#
#     gbm = xgb.XGBClassifier(**{'n_jobs': n_jobs, 'random_state': 1, 'n_estimators': 100,
#                                'importance_type': 'gain', 'use_label_encoder': False})
#     gbm.fit(train_x_new, train_y.values.ravel())
#     new_features = []
#     for feature, imp in zip(train_x_new.columns, gbm.feature_importances_):
#         new_features.append([feature, imp])
#     new_features = sorted(new_features, key=lambda x:x[1], reverse=True)
#     new_features = [item[0] for item in new_features[:top]]
#     print("The number of new features", len(new_features))
#     final_auc = get_results_by_new_features(new_features, train_x.csv, test_x, train_y, test_y, n_jobs)
#     print("The final auc is", final_auc)