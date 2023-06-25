from cmath import nan
import numpy as np
import pandas as pd

all_operators = ["freq"]
num_operators = ["abs", "log", "sqrt", "square", "sigmoid", "round", "residual"]
num_num_operators = ["min", "max", "+", "-", "*", "/"]
cat_num_operators = ["GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
                     "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank"]
cat_cat_operators = ["Combine", "CombineThenFreq", "GroupByThenNUnique"]

symmetry_operators = ["min", "max", "+", "-", "*", "/", "Combine", "CombineThenFreq"]
cal_all_operators = ["freq",
                     "GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
                     "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank",
                     "Combine", "CombineThenFreq", "GroupByThenNUnique"]
# 有必要把numerical中的discrete单独划分出来吗（ordinal features）？
# 感觉是有必要的，这些特征既可以当做numerical也可以当做categorical
# ordinal features可以出现在任何运算符的任何位置


def _reduce_memory(df):
    if str(df.dtypes) in ['object', 'category']:
        return df
    cmin = df.min()
    cmax = df.max()
    if str(df.dtypes)[:3] == 'int':
        # Can use unsigned int here too
        if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
            df = df.astype(np.int8)
        elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
            df = df.astype(np.int16)
        elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
            df = df.astype(np.int32)
        elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
            df = df.astype(np.int64)
    else:
        if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
            df = df.astype(np.float16)
        elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
            df = df.astype(np.float32)
        else:
            df = df.astype(np.float64)
    return df


class Node(object):
    def __init__(self, op, children):
        self.name = op
        self.children = children
        self.data = None
        self.train_idx = []
        self.val_idx = []

    def get_fnode(self):
        fnode_list = []
        for child in self.children:
            fnode_list.extend(child.get_fnode())
        return fnode_list

    def delete(self):
        self.data = None
        for child in self.children:
            child.delete()

    def f_delete(self):
        for child in self.children:
            child.f_delete()

    def calculate(self, data, is_root=False):
        # update: 0: 用新数据计算，可以作为自动的初始化。 1：update，但是当children有group_by的时候更高阶的都要全部计算

        if self.name in all_operators+num_operators:
            d = self.children[0].calculate(data)
            if self.name == "abs":
                new_data = d.abs()
            elif self.name == "log":
                new_data = np.log(np.abs(d))
            elif self.name == "sqrt":
                new_data = np.sqrt(np.abs(d))
            elif self.name == "square":
                new_data = np.square(d)
            elif self.name == "sigmoid":
                new_data = 1 / (1 + np.exp(-d))
            elif self.name == "freq":
                value_counts = d.value_counts()
                value_counts.loc[np.nan] = np.nan
                new_data = d.apply(lambda x: value_counts.loc[x]) # 如果category是int，就必须用.loc[]而非[]
            elif self.name == "round":
                new_data = np.floor(d)
            elif self.name == "residual":
                new_data = d - np.floor(d)
            else:
                raise NotImplementedError(f"Unrecognize operator {self.name}.")
        elif self.name in num_num_operators:
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            if self.name == "max":
                new_data = np.maximum(d1, d2)
            elif self.name == "min":
                new_data = np.minimum(d1, d2)
            elif self.name == "+":
                new_data = d1 + d2
            elif self.name == "-":
                new_data = d1 - d2
            elif self.name == "*":
                new_data = d1 * d2
            elif self.name == "/":
                new_data = d1 / d2.replace(0, np.nan)
        else:
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            if self.name == "GroupByThenMin":
                temp = d1.groupby(d2).min()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMax":
                temp = d1.groupby(d2).max()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMean":
                temp = d1.groupby(d2).mean()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMedian":
                temp = d1.groupby(d2).median()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenStd":
                temp = d1.groupby(d2).std()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == 'GroupByThenRank':
                new_data = d1.groupby(d2).rank(ascending=True, pct=True)
            elif self.name == "GroupByThenFreq":
                def _f(x):
                    value_counts = x.value_counts()
                    value_counts.loc[np.nan] = np.nan
                    return x.apply(lambda x: value_counts.loc[x])
                new_data = d1.groupby(d2).apply(_f)
            elif self.name == "GroupByThenNUnique":
                nunique = d1.groupby(d2).nunique()
                nunique.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: nunique.loc[x])
            elif self.name == "Combine":
                temp = d1.astype(str) + '_' + d2.astype(str)
                temp[d1.isna() | d2.isna()] = np.nan
                temp, _ = temp.factorize()
                new_data = pd.Series(temp, index=d1.index).astype("float64")
            elif self.name == "CombineThenFreq":
                temp = d1.astype(str) + '_' + d2.astype(str)
                temp[d1.isna() | d2.isna()] = np.nan
                value_counts = temp.value_counts()
                value_counts.loc[np.nan] = np.nan
                new_data = temp.apply(lambda x: value_counts.loc[x])
            else:
                raise NotImplementedError(f"Unrecognized operator {self.name}.")
        if self.name == 'Combine':
            new_data = new_data.astype('category')
        else:
            new_data = new_data.astype('float')
            # new_data = new_data.replace([np.inf, -np.inf], np.nan)
        if is_root:
            self.data = new_data
            # self.data = _reduce_memory(self.data)
        return new_data



class FNode(object):
    def __init__(self, name):
        self.name = name
        self.data = None
        self.calculate_all = False

    def delete(self):
        self.data = None

    def f_delete(self):
        self.data = None

    def get_fnode(self):
        return [self.name]

    def calculate(self, data):
        self.data = data[self.name]
        return self.data