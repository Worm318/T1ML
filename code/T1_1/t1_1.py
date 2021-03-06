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

#print df
#print istest


print "\nParte (b)"

print "Shape"
df.shape
print df.shape

print "Info"
df.info()
print "\nDescribe"
df.describe()
print df.describe()


print "\nParte (c)"
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']

print df_scaled.describe()

#print df
#print df_scaled


print "Parte (d)"
import sklearn.linear_model as lm

#le saca la ultima columna de lpsa (?) (no es lo que realmente vamos a entrenar (?))
X = df_scaled.ix[:,:-1]
N = X.shape[0]

#agrega una ultima columna intercept con puros unos (?)
X.insert(X.shape[1], 'intercept', np.ones(N))

y = df_scaled['lpsa']
Xtrain = X[istrain]

#print "Xtrain"
#print Xtrain.describe()

ytrain = y[istrain]

#print "ytrain2"
#print ytrain.describe()

Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
#crea automagicamente el modelo, fit_intercept False por razones
linreg = lm.LinearRegression(fit_intercept = False)

linreg.fit(Xtrain, ytrain)

print "Parte (e)"

#coeficientes
coef = linreg.coef_

XF = Xtrain

#estimamos standard-deviation
yf = ytrain.as_matrix()
XF = XF.as_matrix() 
XFT = np.transpose(XF)
XTXi = np.matrix(np.dot(XFT,XF)).getI() #(XTX)-1

N = XF.shape[0]
p = XF.shape[1]
print "N:"+str(N)+" p:"+str(p)

std = ytrain - linreg.predict(Xtrain)
std = np.sqrt( (1./(N-p-1)) * np.sum(np.square(std)))
print "Standard Deviation"
print std

zscore = coef / np.matrix(np.sqrt(np.diag(XTXi)))
zscore = zscore / std

print "Coeficientes"
print coef
print "Z-Scores"
print zscore

print "confianza"
s = np.random.standard_t(df=N-p,size=100000)
zsig = np.percentile(s,97.5)
print zscore[ zscore > zsig ]
	
s = np.random.normal(size=100000)
zsig = np.percentile(s,97.5)
print zscore[ zscore > zsig ]

print "Parte (f)"

yhat_test = linreg.predict(Xtest)
mse_test = np.mean(np.power(linreg.predict(Xtest) - ytest, 2))


from sklearn import cross_validation
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
k_fold = cross_validation.KFold(len(Xm),10)
mse_cv = 0
mse_cv_train = 0
min_mse = float("Inf")
min_mse_test = float("Inf")
for k, (train, val) in enumerate(k_fold):
	linreg = lm.LinearRegression(fit_intercept = False)
	linreg.fit(Xm[train], ym[train])
	yhat_val = linreg.predict(Xm[val])
	mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
	mse_cv += mse_fold
	if min_mse >= mse_fold : min_mse = mse_fold
	mse_cv_train += np.mean(np.power(linreg.predict(Xtest) - ytest,2))
	if min_mse_test >= np.mean(np.power(linreg.predict(Xtest) - ytest,2)) : min_mse_test = np.mean(np.power(linreg.predict(Xtest) - ytest,2))
mse_cv = mse_cv / 10
mse_cv_train = mse_cv_train / 10

print "Error 10-fold cross-validation"
print "Mean fold MSE(val): %f"%mse_cv
print "Mean fold MSE(test): %f"%mse_cv_train
print "Min MSE(val): %f"%min_mse
print "Min MSE(test): %f"%min_mse_test

k_fold = cross_validation.KFold(len(Xm),5)
mse_cv = 0
mse_cv_train = 0
min_mse = float("Inf")
min_mse_test = float("Inf")
for k, (train, val) in enumerate(k_fold):
	linreg = lm.LinearRegression(fit_intercept = False)
	linreg.fit(Xm[train], ym[train])
	yhat_val = linreg.predict(Xm[val])
	mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
	mse_cv += mse_fold
	if min_mse >= mse_fold : min_mse = mse_fold
	mse_cv_train += np.mean(np.power(linreg.predict(Xtest) - ytest,2))
	if min_mse_test >= np.mean(np.power(linreg.predict(Xtest) - ytest,2)) : min_mse_test = np.mean(np.power(linreg.predict(Xtest) - ytest,2))
mse_cv = mse_cv / 5
mse_cv_train = mse_cv_train / 5

print "Error 5-fold cross-validation"
print "Mean fold MSE(val): %f"%mse_cv
print "Mean fold MSE(test): %f"%mse_cv_train
print "Min MSE(val): %f"%min_mse
print "Min MSE(test): %f"%min_mse_test

print "Error full linear regresion"
print mse_test

