import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

# metodo del codo para k optima
def metodo_codo(datos, k_max=10, random_state=67, ax=None):
    inercias = []
    k_valores = range(1, k_max + 1)
    
    for k in k_valores:
        kmedias = KMeans(n_clusters=k, init='random', random_state=random_state, n_init=10)
        kmedias.fit(datos)
        inercias.append(kmedias.inertia_)  # suma de distancias^2 al centroide más cercano 
    
    if ax is None: # AGREGADO
        ax = plt.gca() # AGREGADO
    
    ax.plot(k_valores, inercias, marker='o')
    ax.set_xlabel('Número de clusters (k)')
    ax.set_ylabel('Inercia (suma de distancias cuadráticas)')
    ax.set_title('Método del codo')
    ax.set_xticks(k_valores)
    ax.grid(True, alpha=0.3)
    
    return inercias

datos = np.loadtxt("iris81_trn.csv", delimiter=',')
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
    ) # Silhouette

    resultados_calinski.append(
        metrics.calinski_harabasz_score(x_iris, etiquetas)
    ) # Calinski harabasz

    resultados_davies.append(
        metrics.davies_bouldin_score(x_iris, etiquetas)
    ) # Davies bouldin

    resultados_fowlkes.append(
        metrics.fowlkes_mallows_score(etiquetas_reales, etiquetas)
    ) # Fowlkes mallows

    resultados_inercia.append(
        kmedias.inertia_
    ) # Inercia

    print(f"\nk = {valor_k}")
    print(f"  Silhouette:        {resultados_silhouette[-1]:.4f}")
    print(f"  Calinski-Harabasz: {resultados_calinski[-1]:.4f}")
    print(f"  Davies-Bouldin:    {resultados_davies[-1]:.4f}")
    print(f"  Fowlkes-Mallows:   {resultados_fowlkes[-1]:.4f}")
    print(f"  Inercia:           {resultados_inercia[-1]:.4f}")


fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Grafico Silhouette
axs[0,0].set_xlabel('k: Numero de clusters')
axs[0,0].set_ylabel('Silhouette')
axs[0,0].plot(k, resultados_silhouette, label='Silhouette', color='blue')
axs[0,0].scatter(k, resultados_silhouette, s=10, color='blue')
axs[0,0].grid()
axs[0,0].legend()

# Grafico Calinski-Harabasz
axs[0,1].set_xlabel('k: Numero de clusters')
axs[0,1].set_ylabel('Calinski-Harabasz')
axs[0,1].plot(k, resultados_calinski, label='Calinski-Harabasz', color='orange')
axs[0,1].scatter(k, resultados_calinski, s=10, color='orange')
axs[0,1].grid()
axs[0,1].legend()

# Grafico Davies-Bouldin
axs[1,0].set_xlabel('k: Numero de clusters')
axs[1,0].set_ylabel('Davies-Bouldin')
axs[1,0].plot(k, resultados_davies, label='Davies-Bouldin', color='green')
axs[1,0].scatter(k, resultados_davies, s=10, color='green')
axs[1,0].grid()
axs[1,0].legend()

# Grafico Fowlkes-Mallows
axs[1,1].set_xlabel('k: Numero de clusters')
axs[1,1].set_ylabel('Fowlkes-Mallows')
axs[1,1].plot(k, resultados_fowlkes, label='Fowlkes-Mallows', color='red')
axs[1,1].scatter(k, resultados_fowlkes, s=10, color='red')
axs[1,1].grid()
axs[1,1].legend()

# Grafico Inercia
axs[2,0].set_xlabel('k: Numero de clusters')
axs[2,0].set_ylabel('Inercia')
axs[2,0].plot(k, resultados_inercia, label='Inercia', color='purple')
axs[2,0].scatter(k, resultados_inercia, s=10, color='purple')
axs[2,0].grid()
axs[2,0].legend()

# Metodo del codo
metodo_codo(x_iris, k_max=10, ax=axs[2,1]) # AGREGADO

plt.tight_layout()
plt.show()
