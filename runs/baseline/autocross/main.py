from AutoCross import *
import multiprocessing as mp
import os, random, pickle

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r


def get_model(name, shape):
    return LogisticRegressionModel(name, shape)
def get_trainer(model):
    return LogisticRegressionTrainer(model)
def next_func(x):
    return x*5 if x<25 else -1


if __name__=='__main__':
    ctx = mp.get_context('spawn') # parallel torch needs 'spawn' $@^@#$^$%*&%@^$%%@
    pool = ctx.Pool(8) # Set n_jobs

    df = pd.DataFrame()
    feats = ['a','b','c']
    for c in feats:
        df[c] = np.random.random(500)
    y = np.array([1 if (df['a'][i]>0.3 and df['b'][i]>0.3) else 0 for i in range(500)])
    label = pd.DataFrame(y)
    print(df)
    print(label)

    perm = np.random.permutation(df.shape[0])
    idxT = perm[:df.shape[0]//2]
    idxV = perm[df.shape[0]//2:]
    idxT.sort()
    idxV.sort()

    discretization = MultiGranularityDiscretization(min_glt=5, next_func=next_func)
    crypto = Crypto(special_char='$')
    featgenerator = FeatGenerator(order=2)
    hashingtrick = HashingTrick(mod=11)
    indexmanager = IndexManager(idx_train_iloc=idxT, idx_valid_iloc=idxV, min_block=128)
    featevaluator = FeatEvaluator(get_model=get_model, get_trainer=get_trainer, hashingtrick=hashingtrick, 
                                indexmanager=indexmanager)
    autocross = AutoCross(df=df, label=label, max_rounds=20, pool = pool,  # pool=None means single thread
                        discretization=discretization, crypto=crypto, featgenerator=featgenerator, featevaluator=featevaluator)

    print(f'Finished! The features are {list(autocross.df.columns)}.')

    if not os.path.exists('cache'): os.makedirs('cache')

    save_variable(autocross,'cache/autocross.pkl')

    df_test = pd.DataFrame()
    for c in feats:
        df_test[c] = np.random.random(500)
    
    autocross.prepare_model()
    pred = autocross.predict(df_test)
    save_variable(pred,'cache/pred.pkl')
    print(pred.shape, pred)

