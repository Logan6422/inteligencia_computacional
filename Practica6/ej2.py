import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import clase_core as ev
import random
import time
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.model_selection import cross_val_score as cvs

# Importamos los datos:
datos = np.loadtxt("./Guia6/leukemia_train.csv", delimiter=',')
y = datos[:,-1]
x = datos[:,:-1]

nro_poblacion = 50
pc = 0.8
pm = 0.01
cant_generaciones = 10

# Inicializamos la población
poblacion = []
for i in range(nro_poblacion):
    # poblacion.append([random.randint(0,1) for _ in range(7129)])
    poblacion.append([1 if random.random() < 0.01 else 0 for _ in range(7129)])

def fitness(cromosoma):
    kn = knc(n_neighbors=5)
    precision = cvs(kn, x[:,cromosoma], y, cv=5).mean()
    cant_trues = np.sum(cromosoma)
    return max((precision-cant_trues)/7129, 0.001)

def selec_padres(poblacion, fitness):
        padres = random.choices(poblacion, weights=fitness, k=2);
        return padres;

def crossover(padre1, padre2, pc):
    if random.random() < pc:
        punto = random.randint(1, len(padre1) - 1);
        hijo1 = padre1[:punto] + padre2[punto:];
        hijo2 = padre2[:punto] + padre1[punto:];
        return hijo1, hijo2
    
    return padre1.copy(), padre2.copy();

def mutacion(cromosoma, pm): #prob mutar
    cromosoma = cromosoma.copy();
    for i in range(len(cromosoma)):
        if random.random() < pm:
            cromosoma[i] = 1 - cromosoma[i]; #mutacion aleatoria
    
    return cromosoma;

def evaluar_poblacion(poblacion):
    resultados = [];
    for cromosoma in poblacion:
        valor = fitness(cromosoma)
        resultados.append(valor);

    return resultados

for i in range(cant_generaciones):
    print(i)
    nueva_poblacion = []

    resultados = evaluar_poblacion(poblacion)
    mejor = np.argmax(resultados)
    cromo_mejor = poblacion[mejor]

    nueva_poblacion.append(cromo_mejor)

    while(len(nueva_poblacion)<nro_poblacion):
        p1,p2 = selec_padres(poblacion,resultados)
        h1,h2 = crossover(p1,p2,pc)
        h1 = mutacion(h1,pm)
        nueva_poblacion.append(h1)
        if (len(nueva_poblacion)<nro_poblacion):
            h2 = mutacion(h2,pm)
            nueva_poblacion.append(h2)
        # print(len(nueva_poblacion))

    poblacion = nueva_poblacion

resultados_finales = evaluar_poblacion(poblacion)
mejor = np.argmax(resultados)
cromo_mejor = poblacion[mejor]

print(f'Mejor cromosoma: {cromo_mejor}')

# Test
datos_tst = np.loadtxt("./Guia6/leukemia_train.csv", delimiter=',')
y_tst = datos_tst[:,-1]
x_tst = datos_tst[:,:-1]

kn2 = knc(n_neighbors=5)
kn2.fit(x[:,cromo_mejor], y)

print("Testeo finalazo:")
prediccion = kn2.predict(x_tst[cromo_mejor])

suma = 0
for i in range(x_tst.shape[0]):
    suma += (prediccion[i] == y_tst[i])

aciertos = suma/x_tst.shape[0]

print(f'Aciertos: {aciertos}')