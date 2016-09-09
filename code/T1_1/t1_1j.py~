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

residuals = linreg.predict(Xtrain) - ytrain

import statsmodels.api as sm
from matplotlib import pyplot as plt

fig = sm.qqplot(residuals,fit=True,line='45')
plt.show()



