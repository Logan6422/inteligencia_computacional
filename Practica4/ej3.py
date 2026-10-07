import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

# TODO: aprenderse lo q hace cada métrica

datos = np.loadtxt("./Guia2/iris81_trn.csv", delimiter=',')
x_iris = datos[:,[0,1,2,3]]
y_iris = datos[:,[4,5,6]]

# Asignación de etiquetas a cada salida de patrón:
# Setosa = 0 (azul)
# Versicolor = 1 (naranja)
# Virginica = 2 (verde)
etiquetas_reales = []
for i in range(y_iris.shape[0]):
    if (y_iris[i,:] == [1,-1,-1]).all():
        etiquetas_reales.append(0)
    elif (y_iris[i,:] == [-1,1,-1]).all():
        etiquetas_reales.append(1)
    else:
        etiquetas_reales.append(2)

k = [2,3,4,5,6,7,8,9,10]
resultados_metodo = []
for i in k:
    kmedias = KMeans(n_clusters=i,
                     init='random',
                     random_state=67)
    kmedias.fit(x_iris)
    etiquetas = kmedias.labels_
    # resultados_metodo.append(
    #     metrics.silhouette_score(x_iris, etiquetas, metric='euclidean'))
    # resultados_metodo.append(
        # metrics.calinski_harabasz_score(x_iris,etiquetas))
    # resultados_metodo.append(
    #         metrics.davies_bouldin_score(x_iris,etiquetas))
    # resultados_metodo.append(
    #     metrics.fowlkes_mallows_score(etiquetas_reales, etiquetas))
    resultados_metodo.append(
        kmedias.inertia_)   # Compactitud

fig, ax = plt.subplots()
ax.set_xlabel('k: Numero de clusters')
ax.plot(k,resultados_metodo)
ax.scatter(k,resultados_metodo, s=10)
plt.show()