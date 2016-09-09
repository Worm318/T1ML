import sklearn.linear_model as lm
import code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bss(x, y, xt, yt, names_x, k = 1):
	p = x.shape[1]-1
	k = min(p, k)
	names_x = np.array(names_x)
	remaining = range(0, p)
	selected = range(0,p+1)
	current_score = 0.0
	best_new_score = 0.0
	results_train = []
	results_test = []
	
	model = lm.LinearRegression(fit_intercept=False)
	model.fit(x,y)
	residuals_train = model.predict(x) - y
	results_train.append(np.mean(np.power(residuals_train, 2)))
	residuals_test = model.predict(xt) - yt
	results_test.append(np.mean(np.power(residuals_test, 2)))

	while len(remaining)>k and len(selected)>k:
		score_candidates = []
		for candidate in remaining:
			#ajustamos el modelo a lo actualmente seleccionado
			model = lm.LinearRegression(fit_intercept=False)
			x_train = x[:,selected]
			model.fit(x_train,y)
			wscore_candidate = abs(model.coef_[selected.index(candidate)])
			
			#removemos la variable y calculamos el training error por removerla
			indexes = selected[:]
			indexes.remove(candidate)
			x_train = x[:,indexes]
			model.fit(x_train, y)
			
			residuals_train = model.predict(x_train) - y
			mse_train_candidate = np.mean(np.power(residuals_train, 2))
			x_test = xt[:,indexes]
			residuals_test = model.predict(x_test) - yt
			mse_test_candidate = np.mean(np.power(residuals_test, 2))

			score_candidates.append((wscore_candidate, candidate, mse_train_candidate, mse_test_candidate))

		score_candidates.sort()
		score_candidates[:] = score_candidates[::-1]
		#code.interact(local=locals())
		print [[round(i[0],4),names_x[i[1]]] for i in score_candidates]
		best_new_score, best_candidate, mse_train, mse_test = score_candidates.pop()
		remaining.remove(best_candidate)
		selected.remove(best_candidate)
		results_train.append(mse_train)
		results_test.append(mse_test)

		print "selected = %s ..."%names_x[best_candidate]
		print "totalvars=%d, abs_weight = %f"%(len(indexes),best_new_score)
		print "mse train=%f test=%f"%(mse_train,mse_test)

	return selected, results_train, results_test



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
selected, result_train, result_test = bss(Xm,ym,Xt,yt,names_regressors)

print selected
for s in selected:
	print names_regressors[s]

print result_train
print len(result_train)
print result_test
print len(result_test)

ax = plt.gca()
ax.invert_xaxis()

plt.xlabel('# Variables')
plt.ylabel('MSE')
plt.plot(range(1,9)[::-1],result_train,'bo')
train_line, = plt.plot(range(1,9)[::-1],result_train,'b-')
plt.plot(range(1,9)[::-1],result_test,'go')
test_line, = plt.plot(range(1,9)[::-1],result_test,'g-')
plt.legend([train_line,test_line],['train error','test error'],loc=1)

plt.show()


