import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.io import mmread
import random

import sklearn.linear_model as lm


print "Cargando set de entrenamiento..."
X = csr_matrix(mmread('train.x.mm'))
y = np.loadtxt('train.y.dat')


print "Cargando set de validacion..."
Xv = csr_matrix(mmread('dev.x.mm'))
yv = np.loadtxt('dev.y.dat')

print "Cargando set de prueba..."
Xt = csr_matrix(mmread('test.x.mm'))
yt = np.loadtxt('test.y.dat')

'''
selected = fss(X,y,Xv,yv)

Xn = X[:,selected]
Xnv = Xv[:,selected]
Xnt = Xt[:,selected]
'''
#alphas_ = np.logspace(100,10,base=10)
#alphas_ = np.logspace(10,0,base=10,num=10)
#alphas_ = np.logspace(3,5,base=10,num=10)
#alphas_ = np.arange(1000.,4000.,500).tolist()
alphas_ = np.arange(1000.,4000.,500).tolist()
model = lm.Lasso(fit_intercept = False)
for a in alphas_:
	print "alpha=%f"%a
	model.set_params(alpha=a)
	model.fit(X,y)
	print "training R2=%f"%model.score(X,y)
	print "validate R2=%f"%model.score(Xv,yv)
	print "test R2=%f"%model.score(Xt,yt)
