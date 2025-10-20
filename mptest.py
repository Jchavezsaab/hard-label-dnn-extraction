import mpmath as mp
import numpy as np
mp.dps = 1000000000000

init=25
n = 64
d = 64
N=1000
alpha=.1

def prop(weights, x):
    if len(weights)==0: return x
    y = x.copy() * weights[0]
    for l in range(1,len(weights)):
        y = leakyReluVec(y)
        y = y * weights[l]
    return y

def compose(weights,x):
    M = weights[0].copy()
    y = x * M
    for l in range(1,len(weights)):
        Mhat = weights[l].copy()
        Mhat = leakyReluMat(Mhat, y)
        M = M*Mhat
        y = y*Mhat
    return M

def leakyReluVec(y):
    res = []
    for yi in y:
        if yi < 0:
            res.append(yi * alpha)
        else:
            res.append(yi)
    return mp.matrix([res])

def leakyReluMat(M,y):
    res = []
    for i in range(n):
        row = []
        for j in range(n):
            if y[i] < 0:
                row.append(M[i,j] * alpha)
            else:
                row.append(M[i,j])
        res.append(row)
    return mp.matrix(res)

def getC(weights, x0, m):
    z = ((weights[0])**(-1))@m
    y = x0*weights[0]
    for i in range(1, len(weights)):
        M = weights[i].copy()
        M = leakyReluMat(M, y)
        z = (M**(-1))*z
        y = y*M
    return z

def pinv(M):
    U,S,V = mp.svd(M)
    S_inv = mp.diag([1/s if abs(s) > mp.eps else 0 for s in S])
    A_pinv = V * S_inv * (U.T)
    return A_pinv

# #LOAD
# import tensorflow as tf
# tf.keras.backend.set_floatx('float64')
# model = tf.keras.models.load_model("data/Unitary_64_64x64_10_float64.keras")
# Nlayers = 0
# weights, biases = [],[]
# for layer in model.layers:
#     if type(layer) == tf.keras.layers.Dense:
#         weights.append(layer.get_weights()[0].astype(np.float64))
#         biases.append(layer.get_weights()[1].astype(np.float64))
#         Nlayers += 1
# print(Nlayers)
# d = Nlayers-1
# n = weights[0].shape[1]

#GENERATE
weights = []
for l in range(d):
    b = []
    for _ in range(n):
        v = np.random.uniform(size=(n,)).astype(np.float64)
        for bi in b:
            v -= np.dot(bi,v)*bi
        b.append(v/np.linalg.norm(v))
    weights.append(np.array(b).T.astype(np.float64))

#NORMALIZE
x0 = np.random.random(size=(N,n)).astype(np.float64)
x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
for l in range(d):
    x0 = x0 @ weights[l]
    scale = np.linalg.norm(x0,axis=1).mean()
    print(scale)
    weights[l] /= scale
    x0 /= scale
    x0[x0 < 0] *= alpha

#CONVERT
weights = [mp.matrix([[mp.mpf(x) for x in row] for row in w]) for w in weights] 

x = np.random.uniform(size=(1,n)).astype(np.float64)
x /= np.linalg.norm(x)
x = mp.matrix([[mp.mpf(xi) for xi in x.flatten().tolist()]])
for l in range(init,d):
#     F = compose(weights[:l], x)
#     y = prop(weights[:l], x)
#     y = leakyReluVec(y)
#     print('y', np.linalg.norm(y))
#     c0 = compose(weights[l:], y)
#     m = compose(weights, x)
#     c = (F**(-1)) * m
#     C = getC(weights[:l], x, m)
#     print(l, np.linalg.matrix_rank(F))
#     print('c0', c0[:2,:2])
#     print('C', C[:2,:2])
#     print('c', c[:2,:2])
    F = compose(weights[:l], x)
    Fi = pinv(F)
    I = Fi * F
    print('I', I[:2,:2])
    print()