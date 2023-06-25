import pandas as pd
import numpy as np
from copy import deepcopy
import random, math
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.tree import _tree, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score



def extract_sklearn(tree):
    tree_ = tree.tree_
    return tree_.threshold[0], tree_.value[tree_.children_left[0]].reshape(-1), tree_.value[tree_.children_right[0]].reshape(-1)


class operator():
    def __init__(self, name, columns=None):
        self.name = name
        if name == 'origin':
            self.n = 1
            self.calc = self.cop
        elif name == '^2':
            self.n = 1
            self.calc = self.pow2
        elif name == '^0.5':
            self.n = 1
            self.calc = self.powhalf
        elif name == '+':
            self.n = 2
            self.calc = self.add
        elif name == '-':
            self.n = 2
            self.calc = self.sub
        elif name == '*':
            self.n = 2
            self.calc = self.mul
        elif name == 'exp(dis)':
            self.n = 2
            self.calc = self.dis
        else:
            print(f'Unknow {name} {columns}')
        self.picks = random.sample(list(columns), self.n)
    
    def cop(self, x): return x
    def pow2(self, x): return x**2
    def powhalf(self, x): return x.abs()**0.5
    def add(self, x, y): return x+y
    def sub(self, x, y): return x-y
    def mul(self, x, y): return x*y
    def dis(self, x, y): return np.exp(-(x-y)**2)
    
    def to_str(self):
        return f"{self.name}({self.picks})"

    def forward(self, df):
        return self.calc(*[df[c] for c in self.picks])

class WeightManager():
    def __init__(self, operates=['^2', '^0.5', '+', '-', '*', 'exp(dis)']):
        self.n = len(operates)
        self.operates = operates
        self.weights = [1/self.n]*self.n
    def sample(self):
        t = random.random()
        idx = 0
        while idx+1<self.n and t>self.weights[idx]: idx+=1
        return self.operates[idx]
    def update(self, op_with_weight):
        mp = {}
        for o in self.operates: mp[o]=[]
        for o, w in op_with_weight: mp[o.name].append(w)
        for i, o in enumerate(self.operates):
            if len(mp[o]):
                self.weights[i] *= math.exp(-1/(1 + sum(mp[o])/len(mp[o])))
        sumw = sum(self.weights)
        self.weights = [w/sumw for w in self.weights]


class TreeNode():
    def __init__(self, df, label, is_regression, wm, E, n_e, min_leaf, return_val=None):
        if return_val is None:
            me = label.values.mean()
            self.return_val = np.array([me] if is_regression else [1-me, me])
        else:
            self.return_val = return_val if is_regression else return_val/return_val.sum()
        if df.shape[0]<=min_leaf or (label[label.columns[0]].nunique == 1):
            self.df, self.label, self.wm = df, label, wm
            self.operator = None
            return
        candidates = []
        eval_func = mutual_info_regression if is_regression else mutual_info_classif
        for i in range(n_e):
            o = operator(wm.sample(), df.columns)
            new_feat = o.forward(df)
            score = eval_func(new_feat.values.reshape(-1,1), label.values)
            candidates.append((o, score))
        wm.update(candidates)

        for c in df.columns:
            o = operator('origin',[c])
            new_feat = o.forward(df)
            score = eval_func(new_feat.values.reshape(-1,1), label.values)
            candidates.append((o, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        while True:
            self.operator = candidates[0][0]
            model = DecisionTreeRegressor if is_regression else DecisionTreeClassifier
            model = model(max_depth=1)
            model.fit(self.operator.forward(df).values.reshape(-1,1), label.values)
            self.threshold, self.lmean, self.rmean = extract_sklearn(model)
            self.df, self.label, self.wm = df, label, wm

            tmp, thr = self.operator.forward(df), self.threshold

            if (tmp<=thr).sum()==0 or (tmp<=thr).sum()==df.shape[0] or candidates[0][1]<1e-3:
                self.operator = None
                del candidates[:1]
                if len(candidates)==0 or candidates[0][1]<1e-3 or True: # !!!!
                    return
            else:
                break
        
        if self.operator.name != 'origin' and (self.operator.to_str() not in [o.to_str() for o in E]):
            E.append(self.operator)

        self.left_child = TreeNode(df[tmp<=thr], label[tmp<=thr], is_regression, deepcopy(wm), E, n_e, min_leaf, self.lmean)
        self.right_child = TreeNode(df[tmp>thr], label[tmp>thr], is_regression, deepcopy(wm), E, n_e, min_leaf, self.rmean)

    def predict(self, df): 
        if self.operator == None:
            return self.return_val
        tmp, thr = self.operator.forward(df), self.threshold
        if tmp.values.reshape(-1)[0]<=thr:
            return self.left_child.predict(df)
        else:
            return self.right_child.predict(df)


class FCTree():
    def __init__(self, df, label, min_leaf, n_e=None, is_regression=True):
        self.E = []
        if n_e is None: n_e=df.shape[1]
        wm = WeightManager()
        self.root = TreeNode(df, label, is_regression, wm, self.E, n_e, min_leaf)
        self.is_regression = is_regression

    def predict(self, df):
        res = []
        for i in df.index:
            t = self.root.predict(df.loc[i:i,:])
            res.append(t[-1])
        return np.array(res)

    def eval_acc(self, df, label):
        pred, y = self.predict(df), label.values
        return ((pred>0.5)*y).mean() + ((pred<=0.5)*(1-y)).mean()
        # return ((y>0.5)*pred).mean() + ((y<=0.5)*(1-pred)).mean()

    def eval_auc(self, df, label):
        pred, y = self.predict(df), label.values
        return roc_auc_score(y, pred)
