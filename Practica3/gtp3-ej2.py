from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np


digits = load_digits()
x, y = digits.data, digits.target
print(f"Datos: {x.shape}, Clases: {len(set(y))}")

x_tr, x_tst, y_tr, y_tst = train_test_split(
    x,
    y,
    test_size = 0.2, # proporcion para test
    # train_size = 0.8, # proporcion para entrenamiento, si no suman 1 scikit da error
    random_state = 67, # semilla aleatoria
    shuffle = True, # mezclar antes de separar, sino toma los primeros N para train y los últimos para test
    stratify = y # mantener proporcion de clases, sino al azar
)
print(f"Entrenamiento = {len(x_tr)}, Test = {len(x_tst)}")

mlp = MLPClassifier(
    hidden_layer_sizes = (64,), # arquitectura
    activation = 'logistic', # f de activacion, relu, tanh, logistic, identity
    solver = 'sgd', # algoritmo de optimizacion, adam, sgd, lbfgs
    max_iter = 1000, # maximo de iteraciones, si no converge antes
    shuffle = True, # mezclar los datos antes de cada epoca
    random_state = 67, # semilla aleatoria, sin esto cada entrenamiento da resultados diferentes
)

# Validacion con 5 folds
kf5 = KFold(
    n_splits = 5, # cantidad de folds
    shuffle = True, # mezclar antes de partir
    random_state = 67 # semilla
)
kfolds5 = []
iteraciones5 = []

NB = GaussianNB()
N_b=[]
iteracionesNB=[]

LDA = LinearDiscriminantAnalysis()
lda=[]
iteracionesLDA=[]

DTREE = DecisionTreeClassifier(
    random_state = 67
)
dt=[]
iteracionesDTREE=[]
KNC = KNeighborsClassifier(
    n_neighbors = 5, # cantidad de vecinos
)
knc=[]
iteracionesKNC=[]
SVC = SVC(
    kernel = 'rbf', # tipo de kernel, linear, poly, rbf, sigmoid
    random_state = 67 # semilla aleatoria
)
svc=[]
iteracionesSVC=[]
for tr_index, tst_index in kf5.split(x):
    x_tr = x[tr_index]
    x_tst = x[tst_index]
    y_tr = y[tr_index]
    y_tst = y[tst_index]

    mlp.fit(x_tr, y_tr) # entrenamiento
    kfolds5.append(mlp.score(x_tst, y_tst)) # mlp[1].score lo hace sin necedidad de predict + accuracy_score
    iteraciones5.append(mlp.n_iter_)

    NB.fit(x_tr, y_tr)
    N_b.append(NB.score(x_tst, y_tst))
    # iteracionesNB.append(NB.n_iter_)

    LDA.fit(x_tr, y_tr)
    lda.append(LDA.score(x_tst, y_tst))
    # iteracionesLDA.append(LDA.n_iter_)

    DTREE.fit(x_tr, y_tr)
    dt.append(DTREE.score(x_tst, y_tst))
    # iteracionesDTREE.append(DTREE.n_iter_)

    KNC.fit(x_tr, y_tr)
    knc_scores = KNC.score(x_tst, y_tst)
    knc.append(knc_scores)
    # iteracionesKNC.append(KNC.n_iter_)

    SVC.fit(x_tr, y_tr)
    svc_scores = SVC.score(x_tst, y_tst)
    svc.append(svc_scores)
    # iteracionesSVC.append(SVC.n_iter_)

media_kfolds5= np.mean(kfolds5)
var_kfolds5 = np.var(kfolds5)

media_NB= np.mean(N_b)
var_NB = np.var(N_b)

media_LDA= np.mean(lda)
var_LDA = np.var(lda)

media_DTREE= np.mean(dt)
var_DTREE = np.var(dt)

media_KNC= np.mean(knc)
var_KNC = np.var(knc)

media_SVC= np.mean(svc)
var_SVC = np.var(svc)

print(f"\nKFolds_5 = {kfolds5}")
print(f"Iteraciones por fold: {iteraciones5}")
print(f"Media KFolds_5: {media_kfolds5}")
print(f"Varianza KFolds_5: {var_kfolds5}")

print(f"\nNB = {N_b}")
print(f"Iteraciones por fold: {iteracionesNB}")
print(f"Media NB: {media_NB:.2%}")
print(f"Varianza NB: {var_NB:.2%}")

print(f"\nLDA = {lda}")
print(f"Iteraciones por fold: {iteracionesLDA}")
print(f"Media LDA: {media_LDA:.2%}")
print(f"Varianza LDA: {var_LDA:.2%}")

print(f"\nDTREE = {dt}")
print(f"Iteraciones por fold: {iteracionesDTREE}")
print(f"Media DTREE: {media_DTREE:.2%}")
print(f"Varianza DTREE: {var_DTREE:.2%}")

print(f"\nKNC = {knc}")
print(f"Iteraciones por fold: {iteracionesKNC}")
print(f"Media KNC: {media_KNC:.2%}")
print(f"Varianza KNC: {var_KNC:.2%}")

print(f"\nSVC = {svc}")
print(f"Iteraciones por fold: {iteracionesSVC}")
print(f"Media SVC: {media_SVC:.2%}")
print(f"Varianza SVC: {var_SVC:.2%}")



