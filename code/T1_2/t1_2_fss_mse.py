import sklearn.linear_model as lm

def fss(x, y, names_x, k = 10000):
	p = x.shape[1]-1
	k = min(p, k)
	names_x = np.array(names_x)
	remaining = range(0, p)
	selected = [p]
	print selected
	current_score = 0.0
	best_new_score = 0.0
	while remaining and len(selected)<=k:
		score_candidates = []
		for candidate in remaining:
			model = lm.LinearRegression(fit_intercept=False)
			indexes = selected + [candidate]
			#print indexes
			#print type(indexes)
			x_train = x[:,indexes]
			predictions_train = model.fit(x_train, y).predict(x_train)
			residuals_train = predictions_train - y
			mse_candidate = np.mean(np.power(residuals_train, 2))
			score_candidates.append((mse_candidate, candidate))
		score_candidates.sort()
		score_candidates[:] = score_candidates[::-1]
		print score_candidates
		best_new_score, best_candidate = score_candidates.pop()
		remaining.remove(best_candidate)
		selected.append(best_candidate)
		print "selected = %s ..."%names_x[best_candidate]
		print "totalvars=%d, mse = %f"%(len(indexes),best_new_score)
	return selected

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

names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45", "Intercept"]
selected = fss(Xm,ym,names_regressors)

print selected
for s in selected:
	print names_regressors[s]
