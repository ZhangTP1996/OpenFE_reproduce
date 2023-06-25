import pandas as pd
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
from sklearn import metrics
import bisect
import os, time
import multiprocessing as mp
torch.set_num_threads(2)

def super_print(*a):
    s = ''
    for x in a: s += str(x)+' '
    print(s)
    os.makedirs('./cache', exist_ok=True)
    open('cache/default_log','a').write(s+'\n')

class Crypto:
    def __init__(self, special_char='$'):
        self.special_char = special_char
    def encode(self, v):
        res = ''
        for i in range(len(v)):
            if i: res += self.special_char
            res += v[i]
        return res
    def decode(self, s):
        return s.split(self.special_char)
    def merge(self, s1, s2):
        v = self.decode(s1) + self.decode(s2)
        v.sort()
        return self.encode(v)

def default_null_next_func(x):
    return -1

class MultiGranularityDiscretization: # 即使全都是category也过一下这个吧，把category都编码成自然数
    def __init__(self, min_glt=10, next_func=default_null_next_func):
        self.min_glt = min_glt
        self.next_func = next_func
        self.map_backup = {}
        self.origin_len = 0
    
    def forward(self, df):
        res = pd.DataFrame()
        cat_columns = df.select_dtypes(exclude=np.number).columns
        num_columns = df.select_dtypes(include=np.number).columns
        self.origin_len = df.shape[0]
        for c in cat_columns:
            v_new, mp = [], {}
            for v in df[c]:
                if pd.isna(v):
                    v_new.append(np.nan)
                else:
                    if v not in mp: mp[v] = len(mp)
                    v_new.append(mp[v])
            series = pd.Series(v_new, dtype=pd.Int64Dtype(), index = df.index)
            res[c] = series
            self.map_backup[c] = mp
        for c in num_columns:
            v_sort = [v for v in df[c] if not pd.isna(v)]
            if len(v_sort) == 0: continue # All Nan
            v_sort.sort()
            i = self.min_glt
            while i>0 and i<=df.shape[0]:
                mp = {}
                for rk in range(len(v_sort)):
                    mp[v_sort[rk]] = int(rk/len(v_sort)*i)
                v_new = [(mp[v] if not pd.isna(v) else np.nan) for v in df[c]]
                series = pd.Series(v_new, dtype=pd.Int64Dtype(), index = df.index)
                res[f'{c}_{i}'] = series
                self.map_backup[f'{c}_{i}'] = mp
                i = self.next_func(i)
        #print("res: {}".format(res.columns.values.tolist()))
        return res
    
    def convert(self, df):
        res = pd.DataFrame()
        cat_columns = df.select_dtypes(exclude=np.number).columns
        num_columns = df.select_dtypes(include=np.number).columns
        for c in cat_columns:
            if c not in self.map_backup:
                super_print(f'Warning : a strange feature {c}')
                continue
            mp = self.map_backup[c]
            v_new = [(mp[v] if v in mp else np.nan) for v in df[c]]
            series = pd.Series(v_new, dtype=pd.Int64Dtype(), index = df.index)
            res[c] = series
        for c in num_columns:
            #if c not in self.map_backup:
            #    super_print(f'Warning : a strange feature {c}')
            #    continue
            i = self.min_glt
            while i>0 and i<=self.origin_len:
                if f'{c}_{i}' not in self.map_backup:
                    super_print(f'Warning : a strange feature {f"{c}_{i}"}')
                    continue
                mp = self.map_backup[f'{c}_{i}']
                v_mp = [o for o in mp] + [-1e100, 1e100]  # !!!!!!!!!!!!!!!
                v_mp.sort()
                v_new = []
                for v in df[c]:
                    if pd.isna(v):
                        v_new.append(np.nan)
                        continue
                    p1 = bisect.bisect_right(v_mp, v) - 1
                    p2 = bisect.bisect_left(v_mp, v)
                    if abs(v-v_mp[p1]) < abs(v-v_mp[p2]):
                        v_new.append(v_mp[p1])
                    else:
                        v_new.append(v_mp[p2])
                v_new = [(mp[v] if v in mp else np.nan) for v in v_new]
                series = pd.Series(v_new, dtype=pd.Int64Dtype(), index = df.index)
                res[f'{c}_{i}'] = series
                i = self.next_func(i)
        #print("res: {}".format(res.columns.values.tolist()))
        return res


class FeatGenerator:
    def __init__(self, order=2):
        if order != 2: raise Exception('Haha! order must be 2 yet :)')
        self.order = 2
        self.map_backup = {}
    
    def forward(self, df, crypto):
        res = pd.DataFrame()
        for c1 in df.columns:
            for c2 in df.columns:
                if c1>=c2 or len(set(crypto.decode(c1)) & set(crypto.decode(c2))): continue
                if crypto.merge(c1, c2) in df.columns: continue
                if crypto.merge(c1, c2) in res.columns: continue
                vec = []
                mp = {}
                for x, y in zip(df[c1], df[c2]):
                    if pd.isna(x) or pd.isna(y):
                        vec.append(np.nan)
                    else:
                        tag = f'{x}x{y}'
                        if tag not in mp: mp[tag]=len(mp)
                        vec.append(mp[tag])
                series = pd.Series(vec, dtype=pd.Int64Dtype(), index = df.index)
                res[crypto.merge(c1, c2)] = series
                self.map_backup[crypto.merge(c1, c2)] = (c1,c2,mp)
        #print("res: {}".format(res.columns.values.tolist()))
        return res
    
    def convert(self, df, new_feats):
        res = df.copy()
        for c in new_feats:
            if c in res.columns: continue
            if c not in self.map_backup:
                super_print(f'Warning : a strange feature {c}')
                continue
            c1, c2, mp = self.map_backup[c]
            vec = []
            for x, y in zip(res[c1], res[c2]):
                if pd.isna(x) or pd.isna(y):
                    vec.append(np.nan)
                else:
                    tag = f'{x}x{y}'
                    vec.append(mp[tag] if tag in mp else np.nan)
            series = pd.Series(vec, dtype=pd.Int64Dtype(), index = df.index)
            res[c] = series
        #print("res: {}".format(res.columns.values.tolist()))
        return res

class LogisticRegressionModel(nn.Module):
    def __init__(self, name, dim):
        super().__init__()
        self.name = name
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim)/dim**0.5)

    def regular(self, l1, l2):
        return self.w.norm(p=1)*l1 + self.w.norm(p=2)*l2

    def forward(self, x):
        output = torch.masked_fill(self.w[x[:,:-1].long()], x[:,:-1]<0, 0).sum(axis=1)
        # output = torch.zeros(x.shape[0])
        output += x[:,-1]
        # for i in range(x.shape[0]):
        #     for j in x[i,:-1]:
        #         if j>=-0.1:
        #             output[i] += self.w[int(j+0.5)]
        return torch.sigmoid(output) # ! tensor *

class LogisticRegressionTrainer:
    def __init__(self, model, batchsize=32, optim=(lambda x: torch.optim.Adam(x.parameters(), lr=1e-3)), l1_reg=1e-4, l2_reg=1e-4,
        loss_func=nn.BCELoss()):
        self.model = model
        self.optim = optim(self.model)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batchsize = batchsize
        self.loss_func = loss_func
    def train_epochs(self, dataset, epochs, debug=False): # dataset : torch TensorDataset(X,y) X:n*m float, y:n float
        self.model.train()
        DL = DataLoader(dataset, self.batchsize, True)
        losses_on_epochs = []
        loss_func = self.loss_func
        for _ in range(epochs):
            losses = []
            for x, y in DL:
                self.optim.zero_grad()
                # if x.shape[1]!=2 or True:
                loss = loss_func(self.model(x),y) + self.model.regular(self.l1_reg, self.l2_reg)
                losses.append(loss.item())
                loss.backward()
                # else:
                #     self.model.w.grad = torch.zeros(len(self.model.w))
                #     loss = 0.
                #     for i in range(len(x)):
                #         thei = int(x[i,0])
                #         if thei < 0: continue
                #         p = 1/(1+torch.exp(-self.model.w[thei]-x[i,-1]))
                #         self.model.w.grad[thei] += (p-1) if y[i]>0.5 else p
                #         loss += float( -torch.log(p) if y[i]>0.5 else -torch.log(1-p) )
                #     self.model.w.grad /= len(x)
                #     regloss = self.model.regular(self.l1_reg, self.l2_reg)
                #     regloss.backward()
                #     losses.append(loss/len(x)+regloss.item())
                self.optim.step()
            losses_on_epochs.append(sum(losses)/len(losses))
        if debug: super_print(f'    Model {self.model.name} trained {len(DL)}x{self.batchsize} with {epochs} epochs.',
              f'loss[0,mid,-1]={[losses_on_epochs[i] for i in [0,epochs//2,-1]]}')
        return losses_on_epochs
    def train_converge(self, dataset, debug=False, validset=None):
        patience = 3
        min_loss = 1e100
        while patience > 0:
            tmp = self.train_epochs(dataset,1,debug)
            o = tmp[-1] if validset==None else self.evalue(validset, debug)
            if min_loss <= o: patience -= 1
            else:
                patience = 3
                min_loss = o
            if patience <=0: break

    def evalue(self, dataset, debug=False):
        self.model.eval()
        DL = DataLoader(dataset, self.batchsize, False)
        loss_func = self.loss_func
        losses = []
        for x, y in DL:
            loss = loss_func(self.model(x),y)
            losses.append(loss.item())
        if debug: super_print(f'  Model {self.model.name} evalued {len(DL)}x{self.batchsize}.',
              f'loss={sum(losses)/len(losses)}')
        return sum(losses)/len(losses)
    
    def predict(self, dataset):
        self.model.eval()
        DL = DataLoader(dataset, self.batchsize, False)
        preds = [self.model(x[0]) for x in DL]
        return torch.cat(preds).detach().numpy()
    def evalue_AUC(self, dataset, debug=False):
        pred = self.predict( TensorDataset(dataset.tensors[0]) )
        y = dataset.tensors[1].numpy().astype('int')
        score = metrics.roc_auc_score(y, pred)
        if debug: super_print(f'  Model {self.model.name} evalue_AUC {y.shape}.', f'AUC={score}')
        return score
    def calc_b(self, X):
        self.model.eval()
        # res = [self.model.w.clone().detach().dot(X[i]) for i in range(X.shape[0])]
        # res = [0]*X.shape[0]
        # for i in range(X.shape[0]):
        #     for j in X[i]:
        #         res[i] += float(self.model.w[int(j+0.5)])
        res = torch.masked_fill(self.model.w.clone().detach()[X.long()], X<0, 0).sum(axis=1)
        return torch.tensor(res).float()

class HashingTrick:
    def __init__(self, mod=41): # 这个数字是随手敲的
        self.mod = mod
        self.mp = {}
    def forward(self, ipt): # ipt can be Series or DataFrame
        df = ipt if 'DataFrame' in str(type(ipt)) else pd.DataFrame(ipt)
        res = []
        base = 0
        if df.shape[1]!=1: print('df.shape', df.shape)
        for c in df.columns:
            mx = df[c].values.max()
            if pd.isna(mx): continue
            dim = min(mx+1,self.mod)
            tmp = torch.tensor(df[c].values).int() # % dim + base
            nan_idx = tmp < 0
            tmp = tmp % dim + base
            tmp[nan_idx] = -1
            res.append(tmp.reshape(-1,1))
            base += dim
        return torch.cat(res, axis=1), base
    def convert(self, ipt, origin):
        res = []
        base = 0
        for c in origin.columns:
            mx = origin[c].values.as_ordered().max()
            if pd.isna(mx): continue
            dim = min(mx+1,self.mod)
            tmp = torch.tensor(ipt[c].values).int() # % dim + base
            nan_idx = tmp < 0
            tmp = tmp % dim + base
            tmp[nan_idx] = -1
            res.append(tmp.reshape(-1,1))
            base += dim
        return torch.cat(res, axis=1), base

class IndexManager:  # Everything in IndexManager is iloc !!!
    def __init__(self, idx_train_iloc, idx_valid_iloc, min_block=1024):
        self.idx_train = idx_train_iloc
        self.idx_valid = idx_valid_iloc
        self.min_block = min_block
    def prepare_successive(self, n_cand):
        self.n_cand = n_cand
        self.start_block = max(self.min_block, len(self.idx_train)*2//n_cand)
    def sample(self, n, m):
        t = np.random.permutation(n)[:m]
        t.sort()
        return t
    def generate_train_index(self, now_cand):
        goal = int(self.n_cand/now_cand*self.start_block)
        if goal <= len(self.idx_train):
            return self.idx_train[self.sample(len(self.idx_train),goal)], 0
        else:
            return self.idx_train, 1
    def generate_valid_index(self, now_cand):
        goal = int(self.n_cand/now_cand*self.start_block)
        if goal <= len(self.idx_valid):
            return self.idx_valid[self.sample(len(self.idx_valid),goal)]
        else:
            return self.idx_valid

import traceback
def A_simple_parallel_work(_):
    try:
        c, trainer, ds_T, ds_V = _ 
        trainer.train_converge(ds_T, validset=ds_V)
        return [trainer.evalue_AUC(ds_V), c]
    except: print(traceback.format_exc())

class FeatEvaluator:
    def __init__(self, get_model, get_trainer, hashingtrick, indexmanager):
        self.get_model = get_model
        self.get_trainer = get_trainer
        self.hashingtrick = hashingtrick
        self.indexmanager = indexmanager

    def build_dataset(self, X, b, y):
        if b is None: b = torch.zeros(X.shape[0])
        if y is None: return TensorDataset( torch.cat([X,b.reshape(X.shape[0],1)],axis=1).float() )
        return TensorDataset( torch.cat([X,b.reshape(X.shape[0],1)],axis=1).float() , torch.tensor(y).float().reshape(-1))

    def pre_train(self, df, y):
        X, width = self.hashingtrick.forward(df)
        X_train, y_train = X[self.indexmanager.idx_train], y[self.indexmanager.idx_train]
        X_valid, y_valid = X[self.indexmanager.idx_valid], y[self.indexmanager.idx_valid] 
        DS = self.build_dataset(X_train, None, y_train)
        self.pre_model = self.get_model(f'OriginModelBeforeSeeking{df.shape[1]+1}-thFeat', width)
        self.pre_trainer = self.get_trainer(self.pre_model)
        self.pre_trainer.train_converge( self.build_dataset(X_train, None, y_train), validset=self.build_dataset(X_valid, None, y_valid), debug=True)
        self.score_org = self.pre_trainer.evalue_AUC( self.build_dataset(X_valid, None, y_valid) )
        return self.pre_trainer.calc_b(X)

    def overall_train(self, df, label):
        X, width = self.hashingtrick.forward(df)
        y = label.values
        self.blocks = 4
        idx = [i for i in range(len(y))]
        self.super_trainers = []
        for i in range(self.blocks):
            l = len(y)*i//self.blocks
            r = len(y)*(i+1)//self.blocks
            idxT = idx[:l] + idx[r:]
            idxV  = idx[l:r]
            model = self.get_model(f'SuperModelWith{df.shape[1]}Feats_{i}', width)
            trainer = self.get_trainer(model)
            trainer.train_converge( self.build_dataset(X[idxT], None, y[idxT]) , debug = False ,
                                 validset=self.build_dataset(X[idxV], None, y[idxV]) )
            self.super_trainers.append(trainer)
    
    def super_predict(self, Ds):
        res = None
        for t in self.super_trainers:
            res = t.predict(Ds) if res is None else res+t.predict(Ds)
        return res/len(self.super_trainers)
    
    def find_best(self, df_org, df_cand, label, pool = None):
        super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'begin find best')
        y = label.values
        Bs = self.pre_train(df_org, y)
        super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'end pre train')

        X_cand = { c : self.hashingtrick.forward(df_cand[c]) for c in df_cand.columns}

        super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'end pool.map hashing trick')
        trainer_cand = {c:self.get_trainer(self.get_model(f'Candidate_{c}', X_cand[c][1])) for c in df_cand.columns}
        self.indexmanager.prepare_successive(df_cand.shape[1])
        iters = 0
        while len(trainer_cand)>1:
            iters += 1
            super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),f'It is {iters}-th round, #cands={len(trainer_cand)}')
            now_i_T, is_all = self.indexmanager.generate_train_index(len(trainer_cand))
            now_i_V = self.indexmanager.generate_valid_index(len(trainer_cand))
            while True:
                if sum(y[now_i_V]) not in [0, len(now_i_V)]: break
                super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),f'now_i_V is bad! they all have same label!')
                now_i_V = self.indexmanager.generate_valid_index(len(trainer_cand))
            
            self.scores = []
            if pool is None:
                for c in trainer_cand:
                    trainer_cand[c].train_converge( self.build_dataset(X_cand[c][0][now_i_T], Bs[now_i_T], y[now_i_T]) )
                    self.scores.append( [trainer_cand[c].evalue_AUC( self.build_dataset(X_cand[c][0][now_i_V], Bs[now_i_V], y[now_i_V]) ), c] )
            else:
                results = pool.map( A_simple_parallel_work, [
                                        (c, trainer_cand[c],
                                        self.build_dataset(X_cand[c][0][now_i_T], Bs[now_i_T], y[now_i_T]),
                                        self.build_dataset(X_cand[c][0][now_i_V], Bs[now_i_V], y[now_i_V]) )
                                            for c in trainer_cand] )
                self.scores = results
            self.scores.sort(key=lambda x:x[0], reverse=True)
            n_remain = 1 if is_all else len(trainer_cand)//2
            for s, c in self.scores[n_remain:]:
                del trainer_cand[c]
                del X_cand[c]
        for c in trainer_cand:
            super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),f'The best feat is {c}. Improve from {self.score_org} to {self.scores[0]}.')
            return c
        raise Exception('Unknow Error #27493272613')

class AutoCross:
    def __init__(self, df, label, max_rounds, discretization, crypto, featgenerator, featevaluator, pool=None, test=None):
        df = discretization.forward(df)
        self.history_scores = []
        self.history_test_scores = []
        patience = 3
        for rounds in range(max_rounds):
            super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f'round {rounds} df_cand = featgenerator.forward(df, crypto)')
            df_cand = featgenerator.forward(df, crypto)
            super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f'round {rounds} best = featevaluator.find_best(df, df_cand, label, pool)')
            best = featevaluator.find_best(df, df_cand, label, pool)
            df[best] = df_cand[best]
            self.history_scores.append(featevaluator.score_org)

            self.df = df
            self.label = label
            self.discretization = discretization
            self.crypto = crypto
            self.featgenerator = featgenerator
            self.featevaluator = featevaluator
            super_print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f'round {rounds} if test is not None')
            if test is not None:
                tmp = self.testAUC(test[0],test[1])
                super_print('Now score =', tmp)
                self.history_test_scores.append(tmp)
            
            if (rounds>2) and (self.history_scores[-1]<self.history_scores[-2]):
                patience -= 1
            else:
                patience = 3
            if patience == 0:
                break
        
        self.df = df
        self.label = label
        self.discretization = discretization
        self.crypto = crypto
        self.featgenerator = featgenerator
        self.featevaluator = featevaluator
    
    def prepare_model(self, n_used_cols=10**10):
        super_print('Start prepare model.')
        self.featevaluator.overall_train(self.df.iloc[:,:n_used_cols], self.label)
    
    def predict(self, df_test, n_used_cols=10**10):
        self.prepare_model(n_used_cols)
        super_print('Start discretization convert.')
        df_test = self.discretization.convert(df_test)
        super_print('Start featgenerator convert.')
        df_test = self.featgenerator.convert(df_test, self.df.columns[:n_used_cols])
        super_print('Start hashingtrick convert.')
        X_test, testwidth = self.featevaluator.hashingtrick.convert(df_test, self.df.iloc[:,:n_used_cols])
        super_print('Start using model to predict.')
        pred = self.featevaluator.super_predict( self.featevaluator.build_dataset(X_test, None, None) )
        return pred
    
    def testAUC(self, df_test, label, n_used_cols=10**10):
        y = label.values
        pred = self.predict(df_test, n_used_cols)
        score = metrics.roc_auc_score(y, pred)
        return score
