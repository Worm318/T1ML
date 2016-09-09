import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.io import mmread
import random

import sklearn.linear_model as lm

from sklearn.decomposition import TruncatedSVD as PCA



print "Cargando set de prueba..."
Xt = csr_matrix(mmread('test.x.mm'))
yt = np.loadtxt('test.y.dat')

def fss(x, y, Xval, yval, k = 1000):
	max_iter = 2000
	p = x.shape[1]-1
	k = min(p, k)
	remaining = range(0, p)
	selected = [p]
	print selected
	current_score = float("Inf")
	best_new_score = float("Inf")
	it = 0
	balancing = 1 #si verdadero mejora entrenamiento y empeora validacion

	while remaining and len(selected)<=k:
		score_candidates = []
		possible_candidates = random.sample(remaining,10)
		for candidate in possible_candidates:
			model = lm.LinearRegression(fit_intercept=False)
			indexes = selected + [candidate]
			x_train = x[:,indexes]
			#print "x:%s x_train:%s"%(type(x),type(x_train))
			model.fit(x_train, y)
			x_val = Xval[:,indexes]
			x_test = Xt[:,indexes]
			score_train = model.score(x_train,y)
			score_val = model.score(x_val,yval)
			score = balancing*(score_train-score_val)
			score_candidates.append((score,score_train,score_val,candidate))
		score_candidates.sort()
		#score_candidates.reverse()
		best_new_score, score_train, score_val, best_candidate = score_candidates.pop()
		remaining.remove(best_candidate)
		selected.append(best_candidate)
		if (balancing > 0):
			if score_train > 0.7 or score_val < 0.2: balancing = -balancing
		else:
			if score_train < 0.2 or score_val > 0.7: balancing = -balancing


		print "selected = %d ..."%best_candidate
		print "totalvars=%d, R2train = %f, R2val = %f"%(len(indexes),score_train,score_val)
		print balancing
		current_score = best_new_score

	return selected

print "Cargando set de entrenamiento..."
X = csc_matrix(mmread('train.x.mm'))
y = np.loadtxt('train.y.dat')


print "Cargando set de validacion..."
Xv = csc_matrix(mmread('dev.x.mm'))
yv = np.loadtxt('dev.y.dat')

selected = fss(X,y,Xv,yv)

Xn = X[:,selected]
Xnv = Xv[:,selected]

model = lm.LinearRegression(fit_intercept = False)
model.fit(Xn,y)

print "training R2=%f"%model.score(Xn,y)
print "validate R2=%f"%model.score(Xnv,yv)



Xnt = Xt[:,selected]
print "test R2=%f"%model.score(Xnt,yt)
