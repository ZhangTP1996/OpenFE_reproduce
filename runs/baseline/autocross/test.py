# import multiprocessing as mp

# def f(_):
#     a, b = _
#     return a+b

# if __name__ == '__main__':
#     print(233)
#     ctx = mp.get_context('spawn')
#     p = ctx.Pool(2)
#     print(p.map(f, [(1,2),(1,2),(1,2),(1,2),(1,2)]))
#     print(p.map(f, [(3,4),(1,2),(1,2),(1,2)]))
#     print(666)

import pickle

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

class A:
    def __init__(self):
        self.v = 1
    def getv(self):
        return self.v

class B:
    def __init__(self, a):
        self.a = a

class C:
    def __init__(self, b):
        self.b = b

a = A()
b = B(a)
c = C(b)

print(c.b.a.getv())

save_variable(c, 'cache/test.pkl')
tmp = load_variavle('cache/test.pkl')

print(tmp.b.a.getv())