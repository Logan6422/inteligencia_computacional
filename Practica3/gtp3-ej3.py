from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import numpy as np

wine = load_wine()
x, y = wine.data, wine.target
print(f"Datos: {x.shape}, Clases: {len(set(y))}")

kf5 = KFold(
    n_splits = 5, # cantidad de folds
    shuffle = True, # mezclar antes de partir
    random_state = 67 # semilla
)

bagging = BaggingClassifier(
    n_estimators = 10, # cantidad de clasificadores
    random_state = 67
)
bagging_scores = []

adaBoost = AdaBoostClassifier(
    n_estimators = 10,
    random_state = 67,
    learning_rate = 1.0
)
adaBoost_scores = []

for tr_index, tst_index in kf5.split(x):
    x_tr = x[tr_index]
    x_tst = x[tst_index]
    y_tr = y[tr_index]
    y_tst = y[tst_index]

    bagging.fit(x_tr, y_tr)
    bagging_scores.append(bagging.score(x_tst, y_tst))

    adaBoost.fit(x_tr, y_tr)
    adaBoost_scores.append(adaBoost.score(x_tst, y_tst))

media_bagging = np.mean(bagging_scores)
var_bagging = np.var(bagging_scores)

media_adaBoost = np.mean(adaBoost_scores)
var_adaBoost = np.var(adaBoost_scores)

print(f"\nBagging = {bagging_scores}")
print(f"Media Bagging: {media_bagging:.2%}")
print(f"Varianza Bagging: {var_bagging:.2%}")

print(f"\nAdaBoost = {adaBoost_scores}")
print(f"Media AdaBoost: {media_adaBoost:.2%}")
print(f"Varianza AdaBoost: {var_adaBoost:.2%}")