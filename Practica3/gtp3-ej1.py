from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

digits = load_digits()
x, y = digits.data, digits.target
print(f"Datos: {x.shape}, Clases: {len(set(y))}")

x_tr, x_tst, y_tr, y_tst = train_test_split(
    x,
    y,
    test_size = 0.2, # proporcion para test
    # train_size = 0.8, # proporcion para entrenamiento, si no suman 1 scikit da error
    random_state = 42, # semilla aleatoria
    shuffle = True, # mezclar antes de separar, sino toma los primeros N para train y los últimos para test
    stratify = y # mantener proporcion de clases, sino al azar
)
print(f"Entrenamiento = {len(x_tr)}, Test = {len(x_tst)}")

mlp = [MLPClassifier(
    hidden_layer_sizes = (100,), # arquitectura
    activation = 'relu', # f de activacion, relu, tanh, logistic, identity
    solver = 'adam', # algoritmo de optimizacion, adam, sgd, lbfgs
    alpha = 0.0001, # parametro de regularizacion, penalizacion de pesos grandes
    batch_size = 'auto', # tamaño de batch, auto = min(200, n_samples)
    learning_rate = 'constant', # tasa de aprendizaje (solo para sgd), constant, invscaling, adaptive
    learning_rate_init = 0.001, # tasa de aprendizaje inicial (solo para sgd y adam)
    max_iter = 1000, # maximo de iteraciones, si no converge antes
    shuffle = True, # mezclar los datos antes de cada epoca
    random_state = None, # semilla aleatoria, sin esto cada entrenamiento da resultados diferentes
    tol = 1e-4, # tolerancia para la convergencia
    verbose = False, # imprimir el progreso de entrenamiento
    warm_start = False, # si True, reutiliza la solucion de la llamada anterior para entrenar mas
    momentum = 0.9, # momento para sgd
    nesterovs_momentum = True, # si True, usa el metodo de Nesterov para sgd
    early_stopping = False, # si True, detiene el entrenamiento cuando la validacion no mejora
    validation_fraction = 0.1, # proporcion de datos para validacion (solo si early_stopping=True)
    beta_1 = 0.9, # parametro de decaimiento para el primer momento (solo para adam)
    beta_2 = 0.999, # parametro de decaimiento para el segundo momento (solo para adam)
    epsilon = 1e-8, # valor para evitar division por cero (solo para adam)
    n_iter_no_change = 10 # numero de iteraciones sin mejora para detener el entrenamiento (solo si early_stopping=True
) for _ in range(3)]

# Particion simple
mlp[0].fit(x_tr, y_tr)
p_simple = mlp[0].score(x_tst, y_tst)
print(f"Particion simple: {p_simple:.4f}")
print(f"Iteraciones realizadas: {mlp[0].n_iter_}")

# Validacion con 5 folds
kf5 = KFold(
    n_splits = 5, # cantidad de folds
    shuffle = True, # mezclar antes de partir
    random_state = 42 # semilla
)
kfolds5 = []
iteraciones5 = []
for tr_index, tst_index in kf5.split(x):
    x_tr = x[tr_index]
    x_tst = x[tst_index]
    y_tr = y[tr_index]
    y_tst = y[tst_index]

    mlp[1].fit(x_tr, y_tr) # entrenamiento
    #y_pred = mlp[1].predict(x_tst) # prediccion
    #accuracy = accuracy_score(y_tst, y_pred) # predicciones correctas / total de muestras
    #kfolds5.append(accuracy)
    kfolds5.append(mlp[1].score(x_tst, y_tst)) # mlp[1].score lo hace sin necedidad de predict + accuracy_score
    iteraciones5.append(mlp[1].n_iter_)

print(f"\nKFolds_5 = {kfolds5}")
print(f"Iteraciones por fold: {iteraciones5}")

media_kfolds5 = np.mean(kfolds5)
var_kfolds5 = np.var(kfolds5)

# Validacion con 10 folds
kf10 = KFold(
    n_splits = 10, # cantidad de folds
    shuffle = True, # mezclar antes de partir
    random_state = 42 # semilla
)
kfolds10 = []
iteraciones10 = []
for tr_index, tst_index in kf10.split(x):
    x_tr = x[tr_index]
    x_tst = x[tst_index]
    y_tr = y[tr_index]
    y_tst = y[tst_index]

    mlp[2].fit(x_tr, y_tr) # entrenamiento
    #y_pred = mlp[2].predict(x_tst) # prediccion
    #accuracy = accuracy_score(y_tst, y_pred) # predicciones correctas / total de muestras
    #folds10.append(accuracy)
    kfolds10.append(mlp[2].score(x_tst, y_tst)) # mlp[2].score lo hace sin necedidad de predict + accuracy_score
    iteraciones10.append(mlp[2].n_iter_)

media_kfolds10 = np.mean(kfolds10)
var_kfolds10 = np.var(kfolds10)
print(f"\nFolds_10 = {kfolds10}")
print(f"Iteraciones por fold: {iteraciones10}")

print(f"\nPartición simple: {p_simple:.2%}")
print(f"5 folds - Media: {media_kfolds5:.2%}, Varianza: {var_kfolds5:.6f}")
print(f"10 folds - Media: {media_kfolds10:.2%}, Varianza: {var_kfolds10:.6f}")