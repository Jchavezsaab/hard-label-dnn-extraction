import numpy as np

n = 64
d = 64
N=1000
alpha=.4

def prop(weights, x):
    if len(weights)==0: return x
    y = x.copy()@ weights[0]
    for l in range(1,len(weights)):
        y[y < 0] *= alpha
        y = y @ weights[l]
    return y

def compose(weights,x):
    M = weights[0].copy()
    y = x.flatten()@M
    for l in range(1,len(weights)):
        Mhat = weights[l].copy()
        Mhat[y < 0,:] *= alpha
        M = M@Mhat
        y = y@Mhat
    return M

def getC(weights, x0, m):
    z = np.linalg.inv(weights[0])@m
    y = x0.flatten()@weights[0]
    for i in range(1, len(weights)):
        M = weights[i].copy()
        M[y < 0] *= alpha
        z = np.linalg.inv(M)@z
        y = y@M
    return z

#GENERATE
weights = []
for l in range(d):
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    weights.append(Q)

# #NORMALIZE
# x0 = np.random.random(size=(N,n)).astype(np.float64)
# x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
# for l in range(d):
#     x0 = x0 @ weights[l]
#     scale = np.linalg.norm(x0,axis=1).mean()
#     print(scale)
#     weights[l] /= scale
#     x0 /= scale
#     x0[x0 < 0] *= alpha

x = np.random.uniform(size=(1,n)).astype(np.float64)
x /= np.linalg.norm(x)
for l in range(1,d):
    F = compose(weights[:l], x)
    print('total conditional',np.linalg.cond(F))
    print('max conditional',max([np.linalg.cond(weights[i]) for i in range(l)]))
    y = prop(weights[:l], x)
    y[y<0] *= alpha
    print('ynorm', np.linalg.norm(y))
    c0 = compose(weights[l:], y)
    m = compose(weights, x)
    c = np.linalg.inv(F) @ m
    C = getC(weights[:l], x, m)
    print(l, np.linalg.matrix_rank(F))
    print(c0[:2,:2])
    print(c[:2,:2])
    print(C[:2,:2])
    print()