import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

# metodo del codo para k optima
def metodo_codo(datos, k_max=10, random_state=67):
    inercias = []
    k_valores = range(1, k_max + 1)
    
    for k in k_valores:
        kmedias = KMeans(n_clusters=k, init='random', random_state=random_state, n_init=10)
        kmedias.fit(datos)
        inercias.append(kmedias.inertia_)  # suma de distancias^2 al centroide más cercano 
    
    plt.figure(figsize=(7,5))
    plt.plot(k_valores, inercias, marker='o')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia (suma de distancias cuadráticas)')
    plt.title('Método del codo')
    plt.xticks(k_valores)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return inercias

datos = np.loadtxt("iris81_trn.csv", delimiter=',')
x_iris = datos[:,[0,1,2,3]]
y_iris = datos[:,[4,5,6]]

inercias = metodo_codo(x_iris, k_max=10)

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
resultados_silhouette = []
resultados_calinski = []
resultados_davies = []
resultados_fowlkes = []
resultados_inercia = []

for valor_k in k:
    kmedias = KMeans(
        n_clusters=valor_k,
        init='random',
        random_state=67,
        n_init=10
    )

    kmedias.fit(x_iris)
    etiquetas = kmedias.labels_
    resultados_silhouette.append(
        metrics.silhouette_score(x_iris, etiquetas, metric='euclidean')
    ) #Silhouette

    resultados_calinski.append(
        metrics.calinski_harabasz_score(x_iris, etiquetas)
    )#Calinski harabasz

    resultados_davies.append(
        metrics.davies_bouldin_score(x_iris, etiquetas)
    ) #Davie bouldin

    resultados_fowlkes.append(
        metrics.fowlkes_mallows_score(etiquetas_reales, etiquetas)
    )#Fowlkes mallows

    resultados_inercia.append(
        kmedias.inertia_
    )#Compactitud / inercia?


fig, ax = plt.subplots()

ax.set_xlabel('k: Numero de clusters')

ax.plot(k, resultados_silhouette, label='Silhouette')
ax.plot(k, resultados_calinski, label='Calinski-Harabasz')
ax.plot(k, resultados_davies, label='Davies-Bouldin')
ax.plot(k, resultados_fowlkes, label='Fowlkes-Mallows')
ax.plot(k, resultados_inercia, label='Inercia')

ax.grid()

ax.scatter(k, resultados_silhouette, s=10)
ax.scatter(k, resultados_calinski, s=10)
ax.scatter(k, resultados_davies, s=10)
ax.scatter(k, resultados_fowlkes, s=10)
ax.scatter(k, resultados_inercia, s=10)

ax.legend()
plt.show()