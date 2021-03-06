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
X.insert(X.shape[1], 'intercept', np.ones(N))

y = df_scaled['lpsa']
Xtrain = X[istrain]
ytrain = y[istrain]

Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]

Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
Xt = Xtest.as_matrix()
yt = ytest.as_matrix()

names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45", "Intercept"]
#selected = fss(Xm,ym,names_regressors)

k=1
p = Xm.shape[1]-1
k = min(p, k)
names_x = np.array(names_regressors)
remaining = range(0, p)
selected = range(0,p+1)
print selected
current_score = 0.0
best_new_score = 0.0
best_test_scores = 0.0
results_train = []
results_test = []

model = lm.LinearRegression(fit_intercept=False)
model.fit(Xm,ym)
residuals_train = model.predict(Xm) - ym
mse_train = np.mean(np.power(residuals_train,2))
residuals_test = model.predict(Xt) - yt
mse_test = np.mean(np.power(residuals_test,2))
results_train.append(mse_train)
results_test.append(mse_test)

while len(remaining)>1 and len(selected)>k:
	score_candidates = []
	for candidate in remaining:
		model = lm.LinearRegression(fit_intercept=False)
		indexes = selected[:]
		indexes.remove(candidate)
		x_train = Xm[:,indexes]

		predictions_train = model.fit(x_train, ym).predict(x_train)
		residuals_train = predictions_train - ym
		mse_candidate = np.mean(np.power(residuals_train, 2))

		residuals_test = model.predict(Xt[:,indexes]) - yt
		mse_test_candidate = np.mean(np.power(residuals_test,2))
		score_candidates.append((mse_candidate, mse_test_candidate, candidate))

	score_candidates.sort()
	score_candidates[:] = score_candidates[::-1]
	print np.round(score_candidates,4)
	best_new_score, best_test_score, best_candidate = score_candidates.pop()
	remaining.remove(best_candidate)
	selected.remove(best_candidate)
	results_train.append(best_new_score)
	results_test.append(best_test_score)

	print "selected = %s ..."%names_x[best_candidate]
	print "totalvars=%d, mse = %f"%(len(indexes),best_new_score)

print np.round(results_train,4)
print np.round(results_test,4)

import matplotlib.pyplot as plt

ax = plt.gca()
ax.invert_xaxis()

plt.xlabel('# Variables')
plt.ylabel('MSE')
plt.plot(range(1,9)[::-1],results_train,'bo')
train_line, = plt.plot(range(1,9)[::-1],results_train,'b-')
plt.plot(range(1,9)[::-1],results_test,'go')
test_line, = plt.plot(range(1,9)[::-1],results_test,'g-')


plt.legend([train_line,test_line],['train error','test error'],loc=1)

plt.show()




