import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import time
import pandas as pd
from collections import Counter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

#algoritmo hungaro para maximizar diagonal de la matriz de contingencia
def mapear_etiquetas(reales, predichas, k):
    cm = contingency_matrix(reales, predichas)   # filas=reales, columnas=predichas
    fila, col = linear_sum_assignment(-cm)        # negamos porque linear_sum_assignment minimiza
    mapeo = {col[i]: fila[i] for i in range(k)}
    return [mapeo[e] for e in predichas]

#metodo del codo para k optima
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

np.random.seed(67)

# Lectura de patrones de entrenamiento
datos = np.loadtxt("iris81_trn.csv", delimiter=',')
x_iris_trn = datos[:,[0,1,2,3]]
y_iris_trn = datos[:,[4,5,6]]

inercias = metodo_codo(x_iris_trn, k_max=10)

# Asignación de etiquetas a cada salida de patrón:
# Setosa = 0 (azul)
# Versicolor = 1 (naranja)
# Virginica = 2 (verde)
etiquetas_datos = []
for i in range(y_iris_trn.shape[0]):
    if (y_iris_trn[i,:] == [1,-1,-1]).all():
        etiquetas_datos.append(0)
    elif (y_iris_trn[i,:] == [-1,1,-1]).all():
        etiquetas_datos.append(1)
    else:
        etiquetas_datos.append(2)

# Para graficación:
# Dimensiones: x=longitud del sépalo (datos[:,0]), y=ancho del pétalo (datos[:,3])
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
colores_datos = {0: 'blue', 1:'orange', 2:'green'}
c1 = [colores_datos[e] for e in etiquetas_datos]

# Figura 1: datos originales
axs[0].scatter(x_iris_trn[:,0], x_iris_trn[:,3], s=10, c=c1, alpha=0.4)
axs[0].set_title('Datos Originales')
axs[0].set_xlim([4,8])
axs[0].set_ylim([0.0,2.6])
axs[0].set_xlabel('Longitud del sépalo')
axs[0].set_ylabel('Ancho del pétalo')


# --------------------------------------------------
# Método K-medias
kmedias = KMeans(n_clusters=3,
                 init='random',
                 random_state=67)

kmedias.fit(x_iris_trn)

centroides_kmedias = kmedias.cluster_centers_
etiquetas_kmedias = kmedias.labels_
# Modificamos los valores de las etiquetas para que coincidan con los asignados de los datos reales
# nuevas_kmedias = {0: 2, 1:0, 2:1}
# nuevas_kmedias = [nuevas_kmedias[e] for e in etiquetas_kmedias]
nuevas_kmedias = mapear_etiquetas(etiquetas_datos, etiquetas_kmedias, k=3)


# Figura 2: Kmeans
# Calculamos porcentaje de aciertos
suma = 0
for i in range(x_iris_trn.shape[0]):
    suma += (nuevas_kmedias[i] == etiquetas_datos[i])
porcentaje_aciertos_kmeans = suma/x_iris_trn.shape[0] * 100

# Graficamos
c2 = [colores_datos[e] for e in nuevas_kmedias]
axs[1].set_title(f'K-medias. Aciertos: {porcentaje_aciertos_kmeans:.2f}')
axs[1].set_xlim([4,8])
axs[1].set_ylim([0.0,2.6])
axs[1].set_xlabel('Longitud del sépalo')
axs[1].scatter(x_iris_trn[:,0], x_iris_trn[:,3], s=13, c=c2, alpha=0.4)
axs[1].scatter(centroides_kmedias[:,0], centroides_kmedias[:,3], s=20, marker='x', c='r')


# --------------------------------------------------
# Método SOM
# Variables
epocasMax_1 = 300
epocasMax_2 = 500
epocasMax_3 = 200
nro_entradas = x_iris_trn.shape[1]
nro_patrones = x_iris_trn.shape[0]
tam_matriz = [3,1]
neuronas = np.empty(shape=(tam_matriz[0],tam_matriz[1],nro_entradas))

# Matriz de posiciones para medir la distancia (entorno de neurona)
neuronas_pos = np.indices((tam_matriz[0], tam_matriz[1])).transpose(1,2,0)

# Inicialización de los pesos (entrada aleatoria)
indices = [0, 1, 2]     # Fijamos los indices para que de siempre el mismo resultado
indice = 0
for i in range(tam_matriz[0]):
    for j in range(tam_matriz[1]):
        neuronas[i,j] = x_iris_trn[indices[indice]]
        indice += 1

# Entrenamiento
# -- Etapa 1
epoca = 0
eta = 0.7
entorno = 3
print("Etapa 1")
inicio = time.perf_counter()
while(epoca<epocasMax_1):
    print(f"Epoca: {epoca}")

    orden = np.random.permutation(nro_patrones)
    for entrada in orden:

        # Seleccion de neurona ganadora
        norma_minima = np.inf
        neuronas_vector = neuronas.reshape(-1, nro_entradas)
        normas = np.subtract(x_iris_trn[entrada], neuronas_vector)
        normas = np.linalg.norm(normas, ord=2, axis=1)
        pos = np.argmin(normas)
        neurona_ganadora = [pos//tam_matriz[1],pos%tam_matriz[1]]

        # Adaptacion de los pesos
        distancias = neuronas_pos - neurona_ganadora
        distancias = np.sum(np.abs(distancias), axis=-1)
        mascara = distancias <= entorno
        errores = np.subtract(x_iris_trn[entrada], neuronas)
        neuronas[mascara] += eta*errores[mascara]

    epoca += 1

# -- Etapa 2
epoca = 0
entorno = np.linspace(entorno, 1, epocasMax_2)
entorno = np.floor(entorno)
# eta = np.linspace(eta, 0.1, epocasMax_2)
eta = np.geomspace(eta, 0.1, epocasMax_2)
print("Etapa 2")
while(epoca<epocasMax_2):
    print(f"Epoca {epoca}")

    orden = np.random.permutation(nro_patrones)
    for entrada in orden:
    # for entrada in range(nro_patrones):
        neurona_ganadora = [0,0]
        norma_minima = np.inf

        # Selección de neurona ganadora (OPTIMIZAR)
        neuronas_vector = neuronas.reshape(-1, nro_entradas)
        normas = np.subtract(x_iris_trn[entrada], neuronas_vector)
        normas = np.linalg.norm(normas, ord=2, axis=1)
        pos = np.argmin(normas)
        neurona_ganadora = [pos//tam_matriz[1],pos%tam_matriz[1]]
        
        # Adaptacion de los pesos
        distancias = neuronas_pos - neurona_ganadora
        distancias = np.sum(np.abs(distancias), axis=-1)
        mascara = distancias <= entorno[epoca]
        errores = np.subtract(x_iris_trn[entrada], neuronas)
        neuronas[mascara] += eta[epoca]*errores[mascara]
    
    epoca += 1

# -- Etapa 3
epoca = 0
entorno = 0
eta = 0.01
print("Etapa 3")
while(epoca<epocasMax_3):
    print(f"Epoca: {epoca}")

    orden = np.random.permutation(nro_patrones)
    for entrada in orden:
    # for entrada in range(nro_patrones):
        neurona_ganadora = [0,0]
        norma_minima = np.inf

        # Seleccion de neurona ganadora
        neuronas_vector = neuronas.reshape(-1, nro_entradas)
        normas = np.subtract(x_iris_trn[entrada], neuronas_vector)
        normas = np.linalg.norm(normas, ord=2, axis=1)
        pos = np.argmin(normas)
        neurona_ganadora = [pos//tam_matriz[1],pos%tam_matriz[1]]

        # Adaptacion de los pesos
        error = x_iris_trn[entrada] - neuronas[neurona_ganadora[0],neurona_ganadora[1]]
        neuronas[neurona_ganadora[0],neurona_ganadora[1]] = neuronas[neurona_ganadora[0],neurona_ganadora[1]] + eta*error
    epoca += 1

# Generamos las etiquetas para cada patrón y la frecuencia de activacion de cada neurona
etiquetas_som = []
frecuencias = np.zeros(neuronas.shape[0])
for entrada in range(x_iris_trn.shape[0]):
    # Seleccion de neurona ganadora
    neuronas_vector = neuronas.reshape(-1, nro_entradas)
    normas = np.subtract(x_iris_trn[entrada], neuronas_vector)
    normas = np.linalg.norm(normas, ord=2, axis=1)
    neurona_ganadora = np.argmin(normas)
    etiquetas_som.append(int(neurona_ganadora))
    frecuencias[neurona_ganadora] += 1

# Modificamos los valores de las etiquetas para que sean igual que etiquetas_datos
# nuevas_som = {0: 2, 1:1, 2:0}
# etiquetas_som = [nuevas_som[e] for e in etiquetas_som]
etiquetas_som = mapear_etiquetas(etiquetas_datos, etiquetas_som, k=3)


# Calculamos porcentaje de aciertos
suma = 0
for i in range(x_iris_trn.shape[0]):
    suma += (etiquetas_som[i] == etiquetas_datos[i])
porcentaje_aciertos_som = suma/x_iris_trn.shape[0] * 100

# Figura 3: SOM
c3 = [colores_datos[e] for e in etiquetas_som]
neuronas = neuronas.squeeze()
centroides_som = neuronas[:, [0, 3]]

axs[2].set_title(f'SOM. Aciertos: {porcentaje_aciertos_som:.2f}')
axs[2].set_xlim([4,8])
axs[2].set_ylim([0.0,2.6])
axs[2].set_xlabel('Longitud del sépalo')
axs[2].scatter(x_iris_trn[:,0], x_iris_trn[:,3], s=13, c=c3, alpha=0.4)
axs[2].scatter(centroides_som[:,0], centroides_som[:,1], s=20, c='r', marker='x')

plt.tight_layout()
plt.show()

# MATRICES DE CONTINGENCIA
print("\nMatriz de Contingencia: Clases Reales vs K-medias")
print(pd.crosstab(etiquetas_datos, nuevas_kmedias, rownames=['Real'], colnames=['K-Medias']))

print("\nMatriz de Contingencia: Clases Reales vs SOM")
print(pd.crosstab(etiquetas_datos, etiquetas_som, rownames=['Real'], colnames=['SOM']))

print("\nMatriz de Contingencia: K-medias vs SOM")
print(pd.crosstab(nuevas_kmedias, etiquetas_som, rownames=['K-Medias'], colnames=['SOM']))



# fig, ax = plt.subplots()
# frecuencias_1d = frecuencias.reshape(-1)  # aplanar a (3,)
# colores = plt.cm.viridis(frecuencias_1d / frecuencias_1d.max())  # mapear frecuencia a color

# ax.bar(range(3), frecuencias_1d, color=colores)
# ax.set_xticks(range(3))
# ax.set_xlabel('Neurona')
# ax.set_ylabel('Frecuencia de activación')

# plt.show()


# GRÁFICO DE NEURONAS DEL SOM EN 2D
# Calcular la clase mayoritaria de Iris para cada neurona original (0, 1, 2)
frecuencias_1d = frecuencias.reshape(-1)  # aplanar a (3,)
clases_neuronas = []
for n in range(3):
    indices_patrones = [i for i, x in enumerate(etiquetas_som) if x == n]
    clases_patrones = [etiquetas_datos[i] for i in indices_patrones]
    if clases_patrones:
        clase_mayoritaria = Counter(clases_patrones).most_common(1)[0][0]
    else:
        clase_mayoritaria = -1
    clases_neuronas.append(clase_mayoritaria)

nombres_clases = ['Setosa', 'Versicolor', 'Virginica']

fig2, ax2 = plt.subplots(figsize=(8, 6))

# Datos de fondo (en gris para que resalten las neuronas)
ax2.scatter(x_iris_trn[:,0], x_iris_trn[:,3], c='lightgray', s=15, alpha=0.5, label='Datos Iris')

# Neuronas - El color representa la frecuencia de activación
sc = ax2.scatter(centroides_som[:,0], centroides_som[:,1], 
                 c=frecuencias_1d, cmap='YlOrRd', s=250, 
                 edgecolors='black', linewidths=1.5, alpha=0.9, zorder=5)
plt.colorbar(sc, label='Frecuencia de activación (cantidad de patrones)')

# Agregar el texto con la clase de Iris mayoritaria dentro de la neurona
for n in range(3):
    if frecuencias_1d[n] > 0:
        ax2.text(centroides_som[n, 0], centroides_som[n, 1], nombres_clases[clases_neuronas[n]], 
                 ha='center', va='center', fontsize=10, fontweight='bold', color='black', zorder=6)

ax2.set_xlabel('Longitud del sépalo')
ax2.set_ylabel('Ancho del pétalo')
ax2.set_xlim([4,8])
ax2.set_ylim([0.0,2.6])
ax2.set_title('Neuronas del SOM en 2D: Frecuencia (color) y Clase Mayoritaria (texto)')
ax2.legend()
plt.tight_layout()
plt.show()