import sys
sys.path.append('../')
from autocross.AutoCross import *
import multiprocessing as mp
np.random.seed(998244353)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_model(name, shape):
    return LogisticRegressionModel(name, shape)


def loss_func(pred, y):
    l = 0.
    for p, v in zip(pred, y):
        if v==1: l-=p.log()
        elif v==0: l-=(1-p).log()*2
        else: print('Error, unknow label',v)
    return l/len(y)


def get_trainer(model):
    return LogisticRegressionTrainer(model, batchsize=32, optim=(lambda x: torch.optim.Adam(x.parameters(), lr=1e-3)),
        l1_reg=1e-4, l2_reg=1e-4, loss_func=loss_func)


def next_func(x):
    return -1

def run_autocross(train_x, val_x, test_x, train_y, val_y, n_jobs, n_new_feats):
    ctx = mp.get_context('spawn')  # parallel torch needs 'spawn' $@^@#$^$%*&%@^$%%@
    pool = ctx.Pool(n_jobs)  # Set n_jobs
    # pool = None
    df = pd.concat([train_x, val_x], axis=0).reset_index(drop=True)
    label = pd.concat([train_y, val_y], axis=0).reset_index(drop=True)
    point = len(train_x.index.tolist())
    idxT = df.index[:point]
    idxV = df.index[point:]
    print("idxT: {}".format(idxT))
    print("idxV: {}".format(idxV))
    discretization = MultiGranularityDiscretization(min_glt=100, next_func=next_func)
    crypto = Crypto(special_char='$')
    featgenerator = FeatGenerator(order=2)
    hashingtrick = HashingTrick(mod=47)
    indexmanager = IndexManager(idx_train_iloc=idxT, idx_valid_iloc=idxV, min_block=16)
    featevaluator = FeatEvaluator(get_model=get_model, get_trainer=get_trainer, hashingtrick=hashingtrick,
                                  indexmanager=indexmanager)
    autocross = AutoCross(df=df, label=label, max_rounds=n_new_feats, pool=pool,  # pool=None means single thread
                          discretization=discretization, crypto=crypto, featgenerator=featgenerator,
                          featevaluator=featevaluator,
                          test=None)
    new_df = pd.concat([df, autocross.df.iloc[:,-n_new_feats:]], axis=1)
    train_x = new_df.iloc[idxT]
    val_x = new_df.iloc[idxV]
    super_print('Start discretization convert.')
    test_x_new = autocross.discretization.convert(test_x)
    super_print('Start featgenerator convert.')
    test_x_new = autocross.featgenerator.convert(test_x_new, autocross.df.columns)
    test_x = pd.concat([test_x, test_x_new.iloc[:,-n_new_feats:]], axis=1)
    for c in train_x.columns[-n_new_feats:]: train_x[c] = train_x[c].astype('category')
    for c in val_x.columns[-n_new_feats:]: val_x[c] = val_x[c].astype('category')
    for c in test_x.columns[-n_new_feats:]: test_x[c] = test_x[c].astype('category')
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
    train_x, val_x, test_x = run_autocross(train_x.loc[train_index], train_x.loc[val_index], test_x,
                                           train_y.loc[train_index], train_y.loc[val_index], test_y)
