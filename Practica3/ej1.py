from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
import numpy as np

#cargo los datos de digitos escritos a mano
datos = load_digits(); #devuelve un Bunch (parecido a diccionario)
x = datos.data; #entradas

#print(datos.data.shape); #datos aplanados
y = datos.target; #salidas deseadas

# #genera una particion con los datos de entrada
# x_train, x_test, y_train, y_test = train_test_split(
#     x,
#     y,
#     test_size=0.2, #particion 80/20
#     random_state=67 #seed
# );

# #Perceptron multiCapa
# red = MLPClassifier(
#     hidden_layer_sizes=(64,), #arquitectura
#     activation='logistic', #funcion de activacion sigmoidea
#     solver='sgd', #algoritmo de ajuste de pesos
#     #stochastic gradient descent
#     max_iter=1000, #iteraciones maximas
#     random_state=67 #seed?
# );
# #entrenar
# red.fit(x_train, y_train);

# #tasas de acierto
# acierto = red.score(x_test, y_test);

# print("Tasa de acierto:", acierto * 100, "%");


#KFOLD 5
kf = KFold(
    n_splits=5, #numero de folds
    shuffle=True, #mezcla los datos antes de dividirlos
    random_state=67 #seed
);

list_acierto = [];
for train_index, test_index in kf.split(x):
    x_train = x[train_index];
    x_test = x[test_index];

    y_train = y[train_index];
    y_test = y[test_index];

    redKFOLD5 = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='logistic',
        solver='sgd',
        max_iter=1000,
        random_state=67
    );

    redKFOLD5.fit(x_train, y_train);
    acierto = redKFOLD5.score(x_test, y_test);

    list_acierto.append(acierto);#guardo los aciertos

media = np.mean(list_acierto);
varianza = np.var(list_acierto)

print("Media k5:", media * 100, "%");
print("Varianza k5:", varianza);

#KFOLD 10
kf10 = KFold(
    n_splits=10, #numero de folds
    shuffle=True, #mezcla los datos antes de dividirlos
    random_state=67 #seed
);

list_acierto = [];
for train_index, test_index in kf10.split(x):
    x_train = x[train_index];
    x_test = x[test_index];

    y_train = y[train_index];
    y_test = y[test_index];

    redKFOLD5 = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='logistic',
        solver='sgd',
        max_iter=1000,
        random_state=67
    );

    redKFOLD5.fit(x_train, y_train);
    acierto = redKFOLD5.score(x_test, y_test);

    list_acierto.append(acierto);#guardo los aciertos

media_k10 = np.mean(list_acierto);
varianza_k10 = np.var(list_acierto)

print("Media k10:", media_k10 * 100, "%");
print("Varianza k10:", varianza_k10);