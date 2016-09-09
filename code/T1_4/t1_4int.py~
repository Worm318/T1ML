import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.io import mmread

import sklearn.linear_model as lm

from sklearn.decomposition import TruncatedSVD as PCA


print "Cargando set de entrenamiento..."
X = csc_matrix(mmread('train.x.mm'))
y = np.loadtxt('train.y.dat')

print "Cargando set de validacion..."
Xv = csr_matrix(mmread('dev.x.mm'))
yv = np.loadtxt('dev.y.dat')

print "Cargando set de prueba..."
Xt = csr_matrix(mmread('test.x.mm'))
yt = np.loadtxt('test.y.dat')

import code
code.interact(local=locals())

'''
fss(X,y,Xv,yv)

model = lm.LinearRegression(fit_intercept = False)
model.fit(X,y)

print "training R2=%f"%model.score(X,y)
print "validate R2=%f"%model.score(Xv,yv)
'''

'''
Full Linear Regression scores
>>> model.score(X,y)
0.99999999888942248
>>> model.score(Xv,yv)
0.61286057101648916
>>> model.score(Xt,yt)
0.59031182165662144
'''
