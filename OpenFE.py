import gc

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from FeatureGenerator import *
from FeatureGenerator import _reduce_memory
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import traceback
from utils import node_to_formula, check_xor, formula_to_node
from sklearn.inspection import permutation_importance
import shap
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, log_loss
import scipy.special
from copy import deepcopy
from tqdm import tqdm



def _enumerate(current_order_num_features, lower_order_num_features,
               current_order_cat_features, lower_order_cat_features):
    num_candidate_features = []
    cat_candidate_features = []
    for op in all_operators:
        for f in current_order_num_features+current_order_cat_features:
            num_candidate_features.append(Node(op, children=[deepcopy(f)]))
    for op in num_operators:
        for f in current_order_num_features:
            num_candidate_features.append(Node(op, children=[deepcopy(f)]))
    for op in num_num_operators:
        for i in range(len(current_order_num_features)):
            f1 = current_order_num_features[i]
            k = i if op in symmetry_operators else 0
            for f2 in current_order_num_features[k:] + lower_order_num_features:
                if check_xor(f1, f2):
                    num_candidate_features.append(Node(op, children=[deepcopy(f1), deepcopy(f2)]))

    for op in cat_num_operators:
        for f in current_order_num_features:
            for cat_f in current_order_cat_features + lower_order_cat_features:
                if check_xor(f, cat_f):
                    num_candidate_features.append(Node(op, children=[deepcopy(f), deepcopy(cat_f)]))
        for f in lower_order_num_features:
            for cat_f in current_order_cat_features:
                if check_xor(f, cat_f):
                    num_candidate_features.append(Node(op, children=[deepcopy(f), deepcopy(cat_f)]))

    for op in cat_cat_operators:
        for i in range(len(current_order_cat_features)):
            f1 = current_order_cat_features[i]
            k = i if op in symmetry_operators else 0
            for f2 in current_order_cat_features[k:] + lower_order_cat_features:
                if check_xor(f1, f2):
                    if op in ['Combine']:
                        cat_candidate_features.append(Node(op, children=[deepcopy(f1), deepcopy(f2)]))
                    else:
                        num_candidate_features.append(Node(op, children=[deepcopy(f1), deepcopy(f2)]))
    return num_candidate_features, cat_candidate_features


def get_candidate_features_high_order(numerical_features_current=None, categorical_features_current=None, ordinal_features_current=None,
                                      numerical_features_lower=None, categorical_features_lower=None, ordinal_features_lower=None):
    if numerical_features_current is None: numerical_features_current = []
    if categorical_features_current is None: categorical_features_current= []
    if ordinal_features_current is None: ordinal_features_current = []
    if numerical_features_lower is None: numerical_features_lower = []
    if categorical_features_lower is None: categorical_features_lower = []
    if ordinal_features_lower is None: ordinal_features_lower = []
    assert len(set(numerical_features_current) & set(categorical_features_current) & set(ordinal_features_current) &\
               set(numerical_features_lower) & set(categorical_features_lower) & set(ordinal_features_lower)) == 0
    # ordinal features既可以当做numerical来计算也可以当做categorical来计算
    current_order_num_features = []
    current_order_cat_features = []
    lower_order_num_features = []
    lower_order_cat_features = []
    for f in numerical_features_current+categorical_features_current+ordinal_features_current:
        if f in ordinal_features_current:
            current_order_num_features.append(FNode(f))
            current_order_cat_features.append(FNode(f))
        elif f in categorical_features_current:
            current_order_cat_features.append(FNode(f))
        else:
            current_order_num_features.append(FNode(f))
    for f in numerical_features_lower+categorical_features_lower+ordinal_features_lower:
        if f in ordinal_features_lower:
            lower_order_num_features.append(FNode(f))
            lower_order_cat_features.append(FNode(f))
        elif f in categorical_features_lower:
            lower_order_cat_features.append(FNode(f))
        else:
            lower_order_num_features.append(FNode(f))

    candidate_features_list = []
    _num, _cat = _enumerate(current_order_num_features, lower_order_num_features,
                            current_order_cat_features, lower_order_cat_features)
    candidate_features_list.extend(_num)
    candidate_features_list.extend(_cat)
    return candidate_features_list

def get_candidate_features(numerical_features=None, categorical_features=None, ordinal_features=None, order=1):
    if numerical_features is None: numerical_features = []
    if categorical_features is None: categorical_features = []
    if ordinal_features is None: ordinal_features = []
    assert len(set(numerical_features) & set(categorical_features) & set(ordinal_features)) == 0
    # ordinal features既可以当做numerical来计算也可以当做categorical来计算
    num_features = []
    cat_features = []
    for f in numerical_features+categorical_features+ordinal_features:
        if f in ordinal_features:
            num_features.append(FNode(f))
            cat_features.append(FNode(f))
        elif f in categorical_features:
            cat_features.append(FNode(f))
        else:
            num_features.append(FNode(f))

    current_order_num_features = num_features
    current_order_cat_features = cat_features
    lower_order_num_features = []
    lower_order_cat_features = []
    candidate_features_list = []

    while order > 0:
        _num, _cat = _enumerate(current_order_num_features, lower_order_num_features,
                                         current_order_cat_features, lower_order_cat_features)
        candidate_features_list.extend(_num)
        candidate_features_list.extend(_cat)
        lower_order_num_features, lower_order_cat_features = current_order_num_features, current_order_cat_features
        current_order_num_features, current_order_cat_features = _num, _cat
        order -= 1
    return candidate_features_list

def _subsample(iterators, fold):
    iterators = list(iterators)
    length = int(len(iterators) / fold)
    random.shuffle(iterators)
    results = [iterators[:length]]
    # iterators = iterators[length:]
    # 1,1,2,4,8,... 份
    while fold != 1:
        fold = int(fold / 2)
        length = int(length * 2)
        if fold == 1:
            results.append(iterators)
        else:
            results.append(iterators[:length])
        # iterators = iterators[length:]
        # length = int(length * 2)
    return results


class OpenFE:
    def __init__(self):
        pass

    def fit(self,
            data, label,
            candidate_features_list,
            train_index, val_index,
            task,
            init_scores=None,
            categorical_features=None,
            metric=None, drop_columns=None,
            fold=64,
            remain=2000, remain_for_stage2=None,
            filter_metric='predictive',
            importance_type='gain_importance',
            stage2_params=None,
            n_repeats=1,
            n_jobs=1,
            seed=1):
        assert importance_type in ['gain_importance', 'permutation', 'shap']
        assert filter_metric in ['predictive', 'corr', 'mi']
        self.data = data
        self.label = label
        self.candidate_features_list = candidate_features_list
        self.train_index = train_index
        self.val_index = val_index
        self.task = task
        self.metric = metric
        self.drop_columns = drop_columns
        self.fold = fold
        self.remain = remain
        self.remain_for_stage2 = remain_for_stage2
        self.filter_metric = filter_metric
        self.importance_type = importance_type
        self.stage2_params = stage2_params
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.seed = seed
        if categorical_features is None:
            self.categorical_features = list(data.select_dtypes(exclude=np.number))
        else:
            self.categorical_features = categorical_features
        np.random.seed(self.seed)
        random.seed(self.seed)
        if init_scores is None:
            print("Start getting initial scores.")
            self.init_scores = self.get_init_score()
        else:
            self.init_scores = init_scores

        print("The number of candidate features", len(self.candidate_features_list))
        self.candidate_features_list = self.stage1_select()
        self.new_features_list = self.stage2_select()

        res = [[node_to_formula(node), score] for node, score in self.new_features_list]
        res = np.array(res)
        np.save('./all_features.npy', res)
        count = 0
        for node, score in self.new_features_list:
            node.delete()
            if score > 0:
                print(count, node_to_formula(node), score)
                count += 1
        gc.collect()
        return self.new_features_list

    def get_init_score(self, use_train=False):
        assert self.task in ["regression", "classification"]
        data = self.data.copy()
        label = self.label.copy()

        params = {"n_estimators": 10000, "learning_rate": 0.1, "metric": self.metric,
                  "seed": self.seed, "n_jobs": self.n_jobs}
        if self.task == "regression":
            gbm = lgb.LGBMRegressor(**params)
        else:
            gbm = lgb.LGBMClassifier(**params)

        for feature in self.categorical_features:
            data[feature] = data[feature].astype('category')
            data[feature] = data[feature].cat.codes
            data[feature] = data[feature].astype('category')

        if self.task == 'classification' and label[label.columns[0]].nunique() > 2:
            oof = np.zeros((len(data), label[label.columns[0]].nunique()))
        else:
            oof = np.zeros(len(data))
        skf = StratifiedKFold(n_splits=5) if self.task == "classification" else KFold(n_splits=5)
        for train_index, val_index in skf.split(data, label):
            X_train, y_train = data.iloc[train_index], label.iloc[train_index]
            X_val, y_val = data.iloc[val_index], label.iloc[val_index]

            gbm.fit(X_train, y_train,
                    eval_set=[[X_val, y_val]], callbacks=[lgb.early_stopping(200)])

            if use_train:
                oof[train_index] += (gbm.predict_proba(X_train, raw_score=True)[:, 1] if self.task == "classification" else \
                                         gbm.predict(X_train)) / (skf.n_splits - 1)
            else:
                oof[val_index] = gbm.predict_proba(X_val, raw_score=True) if self.task == "classification" else \
                    gbm.predict(X_val)

        oof = pd.DataFrame(oof, index=data.index)
        return oof

    def stage1_select(self, ratio=0.5):
        # 提升为全局变量方便multiprocessing
        global _data
        global _label
        global _init_scores
        _data = self.data.copy()
        _label = self.label.copy()
        _init_scores = self.init_scores.copy()

        # 采样成多块数据，每块有1,1,2,4,8...份，每块数据计算之后，去除排序靠后的特征，然后加入更多数据进行计算和evaluation
        train_index_samples = _subsample(self.train_index, self.fold)
        val_index_samples = _subsample(self.val_index, self.fold)
        start = datetime.now()
        idx = 0
        train_idx = train_index_samples[idx]
        val_idx = val_index_samples[idx]
        idx += 1
        results = self._calculate_and_evaluate(self.candidate_features_list, train_idx, val_idx)
        candidate_features_scores = sorted(results, key=lambda x: x[1], reverse=True)
        candidate_features_scores = self.delete_same(candidate_features_scores)
        print(node_to_formula(candidate_features_scores[0][0]))
        print(candidate_features_scores[0][0].data)
        print("Top 20 at", idx)
        print([[node_to_formula(node), score] for node, score in candidate_features_scores[:20]])

        print("Time spent", idx, datetime.now() - start)

        # 两个停止条件，全部样本计算完毕，或者排序收敛
        while idx != len(train_index_samples):
            start = datetime.now()
            # 根据ratio进行deletion，这个ratio可以调整，甚至可以一开始ratio小（保留少），后面调大
            n_reserved_features = max(int(len(candidate_features_scores)*ratio),
                                      min(len(candidate_features_scores), self.remain))
            train_idx = train_index_samples[idx]
            val_idx = val_index_samples[idx]
            idx += 1
            if n_reserved_features <= self.remain:
                # 如果剩余的特征太少了, 提前返回，但是需要计算所有数据为了two_phase_select
                print("Early return at idx", idx)
                train_idx = train_index_samples[-1]
                val_idx = val_index_samples[-1]
                idx = len(train_index_samples)
            else:
                deleted = [[node_to_formula(node), score] for node, score in candidate_features_scores[n_reserved_features:]]
                deleted = np.array(deleted)
                np.save('deleted_%d.npy' % (idx-1), deleted)
            candidate_features_list = [item[0] for item in candidate_features_scores[:n_reserved_features]]
            del candidate_features_scores[n_reserved_features:]; gc.collect()
            print("The number of candidate features", len(candidate_features_list))
            # candidate_features_scores_pre = candidate_features_scores

            results = self._calculate_and_evaluate(candidate_features_list, train_idx, val_idx) # !!!!
            candidate_features_scores = sorted(results, key=lambda x: x[1], reverse=True)
            print("Top 20 at", idx)
            print([[node_to_formula(node), score] for node, score in candidate_features_scores[:20]])

            print("Time spent", idx, datetime.now() - start)
            print(node_to_formula(candidate_features_scores[0][0]))
            print(candidate_features_scores[0][0].data)
            print(node_to_formula(candidate_features_scores[1][0]))
            print(candidate_features_scores[1][0].data)

        print(f'stopped at idx {idx}')
        if self.remain_for_stage2 is not None:
            if len(candidate_features_scores) < self.remain_for_stage2:
                return candidate_features_scores
            elif candidate_features_scores[self.remain_for_stage2][1] <= 0:
                return candidate_features_scores[:self.remain_for_stage2]
            else:
                return [item for item in candidate_features_scores if item[1] > 0]
        else:
            return [item for item in candidate_features_scores if item[1] > 0]

    def stage2_select(self):
        data_new = []
        new_features = []
        for feature, score in self.candidate_features_list:
            new_features.append(node_to_formula(feature))
            data_new.append(feature.data.values)
        data_new = np.vstack(data_new)
        print(data_new.T.shape)
        start = datetime.now()
        data_new = pd.DataFrame(data_new.T, index=self.candidate_features_list[0][0].data.index,
                                columns=['autoFE-%d' % i for i in range(len(new_features))])
        data_new = pd.concat([data_new, self.data], axis=1)
        for f in self.categorical_features:
            data_new[f] = data_new[f].astype('category')
            data_new[f] = data_new[f].cat.codes
            data_new[f] = data_new[f].astype('category')
        data_new = data_new.replace([np.inf, -np.inf], np.nan)
        if self.drop_columns is not None:
            data_new = data_new.drop(self.drop_columns, axis=1)
        train_y = self.label.loc[self.train_index]
        val_y = self.label.loc[self.val_index]
        train_init = self.init_scores.loc[self.train_index]
        val_init = self.init_scores.loc[self.val_index]

        train_x = data_new.loc[self.train_index]
        val_x = data_new.loc[self.val_index]
        if self.stage2_params is None:
            params = {"n_estimators": 1000, "importance_type": "gain", "num_leaves": 16,
                      "seed": 1, "n_jobs": self.n_jobs}
        else:
            params = self.stage2_params
        if self.metric is not None:
            params.update({"metric": self.metric})
        if self.task == 'classification':
            gbm = lgb.LGBMClassifier(**params)
        else:
            gbm = lgb.LGBMRegressor(**params)
        gbm.fit(train_x, train_y, init_score=train_init,
                eval_init_score=[val_init],
                eval_set=[(val_x, val_y)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(1)])
        print("Time spent training phase I", datetime.now() - start)
        init_metric = self.get_init_metric(val_init, val_y)
        key = list(gbm.best_score_['valid_0'].keys())[0]
        score = init_metric - gbm.best_score_['valid_0'][key]
        print(f"The estimated improvement of {self.metric} is {score}")
        results = []
        if self.importance_type == 'gain_importance':
            for i, imp in enumerate(gbm.feature_importances_[:len(new_features)]):
                results.append([formula_to_node(new_features[i]), imp])
        elif self.importance_type == 'permutation':
            r = permutation_importance(gbm, val_x, val_y, scoring='r2',
                                       n_repeats=self.n_repeats, random_state=self.seed, n_jobs=self.n_jobs)
            for i, imp in enumerate(r.importances_mean[:len(new_features)]):
                results.append([formula_to_node(new_features[i]), imp])
        elif self.importance_type == 'shap':
            explainer = shap.TreeExplainer(gbm)
            shap_values = explainer.shap_values(data_new.values)
            shap_values = np.mean(np.abs(shap_values), axis=0)
            for i, imp in enumerate(shap_values[:len(new_features)]):
                results.append([formula_to_node(new_features[i]), imp])
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results

    def transform(self, X_train, X_test, n_new_features):
        print("Start transforming.")
        if n_new_features == 0:
            return X_train, X_test
        _data1 = pd.concat([X_train, X_test], axis=0)
        n_train = len(X_train)
        print(_data1.shape, n_train, self.n_jobs)
        ex = ProcessPoolExecutor(self.n_jobs)
        results = []
        for feature, _ in self.new_features_list[:n_new_features]:
            print(node_to_formula(feature))
            feature = formula_to_node(node_to_formula(feature))
            results.append(ex.submit(self._cal, feature, _data1[feature.get_fnode()].copy(), n_train))
        ex.shutdown(wait=True)
        print("Finish multiprocessing")
        _train = []
        _test = []
        names = []
        names_map = {}
        for i, res in enumerate(results):
            d1, d2, f = res.result()
            names.append('autoFE_f_%d' % i)
            names_map['autoFE_f_%d' % i] = f
            _train.append(d1)
            _test.append(d2)
        print("start concatenating")
        _train = np.vstack(_train)
        _test = np.vstack(_test)
        _train = pd.DataFrame(_train.T, columns=names, index=X_train.index)
        _test = pd.DataFrame(_test.T, columns=names, index=X_test.index)
        _train = pd.concat([X_train, _train], axis=1)
        _test = pd.concat([X_test, _test], axis=1)
        print(_train.shape, _test.shape)
        return _train, _test

    def _cal(self, feature, data_tmp, n_train):
        feature.calculate(data_tmp, is_root=True)
        if (str(feature.data.dtype) == 'category') | (str(feature.data.dtype) == 'object'):
            # factorize, _ = feature.data.factorize()
            # feature.data.values = factorize
            pass
        else:
            feature.data = feature.data.replace([-np.inf, np.inf], np.nan)
            feature.data = feature.data.fillna(0)
        print(node_to_formula(feature))
        return feature.data.values.ravel()[:n_train], \
               feature.data.values.ravel()[n_train:], \
               node_to_formula(feature)

    def get_init_metric(self, pred, label):
        # 要注意metric是越大越好还是学越小越好
        if self.metric == 'binary_logloss':
            init_metric = log_loss(label, scipy.special.expit(pred))
        if self.metric == 'multi_logloss':
            init_metric = log_loss(label, scipy.special.softmax(pred, axis=1))
        if self.metric == 'rmse':
            init_metric = mean_squared_error(label, pred, squared=False)
        return init_metric

    def delete_same(self, candidate_features_scores, threshold=1e-20):
        start_n = len(candidate_features_scores)
        if candidate_features_scores:
            pre_score = candidate_features_scores[0][1]
            pre_feature = node_to_formula(candidate_features_scores[0][0])
        i = 1
        count = 0
        while i < len(candidate_features_scores):
            now_score = candidate_features_scores[i][1]
            now_feature = node_to_formula(candidate_features_scores[i][0])
            if abs(now_score - pre_score) < threshold:
                candidate_features_scores.pop(i)
                if count < 100:
                    print(pre_feature, pre_score, now_feature, now_score)
                    count += 1
            else:
                pre_score = now_score
                pre_feature = now_feature
                i += 1
        end_n = len(candidate_features_scores)
        print("%d same features have been deleted." % (start_n - end_n))
        return candidate_features_scores

    def _evaluate(self, candidate_feature, train_y, val_y, train_init, val_init, init_metric):
        try:
            train_x = pd.DataFrame(candidate_feature.data.loc[train_y.index])
            val_x = pd.DataFrame(candidate_feature.data.loc[val_y.index])
            if len(train_x) != len(train_y):
                print(len(train_x), len(train_y))
            if self.filter_metric == 'predictive':
                params = {"n_estimators": 100, "importance_type": "gain", "num_leaves": 16,
                          "seed": 1, "deterministic": True, "n_jobs": 1}
                if self.metric is not None:
                    params.update({"metric": self.metric})
                if self.task == 'classification':
                    gbm = lgb.LGBMClassifier(**params)
                else:
                    gbm = lgb.LGBMRegressor(**params)
                gbm.fit(train_x, train_y, init_score=train_init,
                        eval_init_score=[val_init],
                        eval_set=[(val_x, val_y)],
                        callbacks=[lgb.early_stopping(3, verbose=False)])
                key = list(gbm.best_score_['valid_0'].keys())[0]
                score = init_metric - gbm.best_score_['valid_0'][key]
            elif self.filter_metric == 'corr':
                score = np.corrcoef(pd.concat([train_x, val_x], axis=0).fillna(0).values.ravel(),
                                    pd.concat([train_y, val_y], axis=0).fillna(0).values.ravel())[0, 1]
                score = abs(score)
            elif self.filter_metric == 'mi':
                r = mutual_info_regression(pd.concat([train_x, val_x], axis=0).fillna(0),
                                           pd.concat([train_y, val_y], axis=0).values.ravel())
                score = r[0]
            else:
                raise NotImplementedError("Cannot recognize filter_metric %s." % self.filter_metric)
        except:
            print(traceback.format_exc())
        return score

    def _calculate_and_evaluate_multiprocess(self, candidate_features, train_idx, val_idx):
        try:
            results = []
            data_temp = _data.loc[train_idx + val_idx]
            train_y = _label.loc[train_idx]
            val_y = _label.loc[val_idx]
            train_init = _init_scores.loc[train_idx]
            val_init = _init_scores.loc[val_idx]
            init_metric = self.get_init_metric(val_init, val_y)
            for candidate_feature in candidate_features:
                candidate_feature.calculate(data_temp, is_root=True)
                score = self._evaluate(candidate_feature, train_y, val_y, train_init, val_init, init_metric)
                results.append([candidate_feature, score])
            return results
        except:
            print(node_to_formula(candidate_feature))
            print(traceback.format_exc())
            exit()

    def _calculate_and_evaluate(self, candidate_features, train_idx, val_idx):
        print("We are using %d data points." % (len(train_idx)+len(val_idx)))
        results = []
        length = int(np.ceil(len(candidate_features) / self.n_jobs / 4))
        # length = min(100, length)
        n = int(np.ceil(len(candidate_features) / length))
        random.shuffle(candidate_features)
        for f in candidate_features:
            f.delete()
        print(self.n_jobs)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            with tqdm(total=n) as progress:
                for i in range(n):
                    if i == (n-1):
                        future = ex.submit(self._calculate_and_evaluate_multiprocess,
                                                 candidate_features[i * length:],
                                                 train_idx, val_idx)
                    else:
                        future = ex.submit(self._calculate_and_evaluate_multiprocess,
                                                 candidate_features[i * length:(i + 1) * length],
                                                 train_idx, val_idx)
                    future.add_done_callback(lambda p: progress.update())
                    results.append(future)
                res = []
                for r in results:
                    res.extend(r.result())
        return res


