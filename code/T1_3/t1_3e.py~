import sklearn.linear_model as lm
import pandas as pd
import numpy as np

url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)

#print df

df = df.drop('Unnamed: 0', axis=1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)
#normalizacion
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']
X = df_scaled.ix[:,:-1]
N = X.shape[0]

#agrega una ultima columna intercept con puros unos (?)
#X.insert(X.shape[1], 'intercept', np.ones(N))

y = df_scaled['lpsa']
Xtrain = X[istrain]
ytrain = y[istrain]

Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]

Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()

from sklearn.linear_model import Ridge,Lasso
import matplotlib.pylab as plt
from sklearn import cross_validation

#X = X.drop('intercept', axis=1)

def MSE(y,yhat): return np.mean(np.power(y-yhat,2))
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
k_fold = cross_validation.KFold(len(Xm),10)

print "\nRidge CV"
best_cv_mse = float("inf")
model = Ridge(fit_intercept=True)
alphas_ = np.logspace(2,-2,base=10)
for a in alphas_:
	print "Testing alpha:%f"%a
	model.set_params(alpha=a)
	mse_list_k10 = [MSE(model.fit(Xm[train], ym[train]).predict(Xm[val]), ym[val]) \
	for train, val in k_fold]
	if np.mean(mse_list_k10) < best_cv_mse:
		#print np.round(mse_list_k10,2)
		best_cv_mse = np.mean(mse_list_k10)
		best_alpha = a
		print "BEST PARAMETER=%f, MSE(CV)=%f"%(best_alpha,best_cv_mse)

		model.set_params(alpha=a)
		model.fit(Xm,ym)
		print "Test Error %f"%MSE(model.predict(Xtest),ytest)

print "\nLasso CV"
best_cv_mse = float("inf")
model = Lasso(fit_intercept=True)
alphas_ = np.logspace(0.5,-2,base=10)
for a in alphas_:
	print "Testing alpha:%f"%a
	model.set_params(alpha=a)
	mse_list_k10 = [MSE(model.fit(Xm[train],ym[train]).predict(Xm[val]), ym[val]) \
	for train, val in k_fold]
	if np.mean(mse_list_k10) < best_cv_mse:
		#print np.round(mse_list_k10,2)
		best_cv_mse = np.mean(mse_list_k10)
		best_alpha = a
		print "BEST PARAMETER=%f, MSE(CV)=%f"%(best_alpha,best_cv_mse)

		model.set_params(alpha=a)
		model.fit(Xm,ym)
		print "Test Error %f"%MSE(model.predict(Xtest),ytest)



