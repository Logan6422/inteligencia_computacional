import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import random
import time

# Carga de datos
datos = np.loadtxt("circulo.csv", delimiter=',')

# Variables
epocasMax_1 = 300
epocasMax_2 = 500
epocasMax_3 = 200
nro_entradas = datos.shape[1]
nro_patrones = datos.shape[0]
tam_matriz = [3,3] #2parte a
# tam_matriz = [1,9] #parte b
neuronas = np.empty(shape=(tam_matriz[0],tam_matriz[1],nro_entradas))

# Matriz de posiciones para medir la distancia (entorno de neurona)
neuronas_pos = np.indices((tam_matriz[0], tam_matriz[1])).transpose(1,2,0)

# Inicialización de los pesos (entrada aleatoria)
#Evita gastar las priemras epocas en acercarse a la nube de puntos
indices = np.arange(nro_patrones)
random.shuffle(indices)
indice = 0
for i in range(tam_matriz[0]):
    for j in range(tam_matriz[1]):
        neuronas[i,j] = datos[indices[indice]]
        indice += 1

# Animación
fig, ax = plt.subplots()

pos_neuronas = []

# Scatter de datos
scat_datos = ax.scatter(datos[:,0], datos[:,1], s=5)

# Scatter de neuronas
x = neuronas[:, :, 0].flatten()
y = neuronas[:, :, 1].flatten()
scat_neuronas = ax.scatter(x,y,s=5,c='r')

# Lineas de neuronas vecinas
def get_segments(neuronas):
    segments = []
    for i in range(tam_matriz[0]):
        for j in range(tam_matriz[1]):
            if j < tam_matriz[1] - 1:  # vecino a la derecha
                segments.append([neuronas[i, j, :2], neuronas[i, j+1, :2]])
            if i < tam_matriz[0] - 1:  # vecino de abajo
                segments.append([neuronas[i, j, :2], neuronas[i+1, j, :2]])
    return segments

lineas = LineCollection(get_segments(neuronas), colors='gray', linewidths=0.5, zorder=1)
ax.add_collection(lineas)

# Extras
texto_frame = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', fontsize=9)
texto_etapa = ax.text(0.02, 0.88, '', transform=ax.transAxes, ha='left', va='top', fontsize=11)
cant_frames = epocasMax_1 + epocasMax_2 + epocasMax_3
ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5))
etapa_actual = 1

def update(frame):
    x = pos_neuronas[frame][:,:,0].flatten()
    y = pos_neuronas[frame][:,:,1].flatten()
    data = np.stack([x,y]).T
    scat_neuronas.set_offsets(data)

    lineas.set_segments(get_segments(pos_neuronas[frame]))

    # Textos
    texto_frame.set_text(f'Frame {frame+1}/{cant_frames}')
    limite1 = epocasMax_1
    limite2 = epocasMax_1 + epocasMax_2
    if frame<limite1:
        etapa_actual = 1
    elif frame<limite2:
        etapa_actual = 2
    else:
        etapa_actual = 3
    texto_etapa.set_text(f'Etapa {etapa_actual}')

    return scat_neuronas

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
        # norma_minima = np.inf
        neuronas_vector = neuronas.reshape(-1, nro_entradas)
        normas = np.subtract(datos[entrada], neuronas_vector)
        normas = np.linalg.norm(normas, ord=2, axis=1)
        pos = np.argmin(normas)
        neurona_ganadora = [pos//tam_matriz[1],pos%tam_matriz[1]]

        # for i in range(tam_matriz[0]):
        #     for j in range(tam_matriz[1]):
        #         norma = np.linalg.norm((datos[entrada]-neuronas[i,j]), ord=2)
        #         if (norma<norma_minima):
        #             norma_minima = norma
        #             neurona_ganadora = [i,j]

        # Adaptacion de los pesos
        distancias = neuronas_pos - neurona_ganadora
        # Distancia manhattan
        distancias = np.sum(np.abs(distancias), axis=-1)
        mascara = distancias <= entorno
        errores = np.subtract(datos[entrada], neuronas)
        neuronas[mascara] += eta*errores[mascara]


        # for i in range(tam_matriz[0]):
        #     for j in range(tam_matriz[1]):
        #         dist = np.subtract(neurona_ganadora,[i,j])
        #         dist = np.sum(abs(dist))
        #         if (dist<=entorno):
        #             error = datos[entrada] - neuronas[i,j]
        #             neuronas[i,j] = neuronas[i,j] + eta*error
    pos_neuronas.append(neuronas.copy())
    # if len(pos_neuronas) > 1:
    #     cambio = np.linalg.norm(pos_neuronas[-1] - pos_neuronas[-2])
    #     print(f"Epoca: {epoca} - cambio: {cambio:.6f}")
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
        # norma_minima = np.inf

        # Selección de neurona ganadora (OPTIMIZAR)
        neuronas_vector = neuronas.reshape(-1, nro_entradas)
        normas = np.subtract(datos[entrada], neuronas_vector)
        normas = np.linalg.norm(normas, ord=2, axis=1)
        pos = np.argmin(normas)
        neurona_ganadora = [pos//tam_matriz[1],pos%tam_matriz[1]]

        # for i in range(tam_matriz[0]):
        #     for j in range(tam_matriz[1]):
        #         norma = np.linalg.norm((datos[entrada]-neuronas[i,j]), ord=2)
        #         if (norma<norma_minima):
        #             norma_minima = norma
        #             neurona_ganadora = [i,j]
        
        # Adaptacion de los pesos
        distancias = neuronas_pos - neurona_ganadora
        distancias = np.sum(np.abs(distancias), axis=-1)
        #EUCLIDEA
        # distancias = neuronas_pos - neurona_ganadora
        # distancias = np.linalg.norm(distancias, ord=2, axis=-1)
        mascara = distancias <= entorno[epoca]
        errores = np.subtract(datos[entrada], neuronas)
        neuronas[mascara] += eta[epoca]*errores[mascara]

        # for i in range(tam_matriz[0]):
        #     for j in range(tam_matriz[1]):
        #         dist = np.subtract(neurona_ganadora,[i,j])
        #         dist = np.sum(abs(dist))
        #         if (dist<=entorno[epoca]):
        #             error = datos[entrada] - neuronas[i,j]
        #             neuronas[i,j] = neuronas[i,j] + eta[epoca]*error
    # print(neuronas[0,0])
    pos_neuronas.append(neuronas.copy())
    epoca += 1

# -- Etapa 3
epoca = 0
entorno = 0
eta = 0.01
print("Etapa 2")
while(epoca<epocasMax_3):
    print(f"Epoca: {epoca}")

    orden = np.random.permutation(nro_patrones)
    for entrada in orden:
    # for entrada in range(nro_patrones):
        neurona_ganadora = [0,0]
        norma_minima = np.inf

        # Seleccion de neurona ganadora (OPTIMIZAR)
        neuronas_vector = neuronas.reshape(-1, nro_entradas)
        normas = np.subtract(datos[entrada], neuronas_vector)
        normas = np.linalg.norm(normas, ord=2, axis=1)
        pos = np.argmin(normas)
        neurona_ganadora = [pos//tam_matriz[1],pos%tam_matriz[1]]

        # for i in range(tam_matriz[0]):
        #     for j in range(tam_matriz[1]):
        #         norma = np.linalg.norm((datos[entrada]-neuronas[i,j]), ord=2)
        #         if (norma<norma_minima):
        #             norma_minima = norma
        #             neurona_ganadora = [i,j]

        # Adaptacion de los pesos
        error = datos[entrada] - neuronas[neurona_ganadora[0],neurona_ganadora[1]]
        neuronas[neurona_ganadora[0],neurona_ganadora[1]] = neuronas[neurona_ganadora[0],neurona_ganadora[1]] + eta*error
    # print(neuronas[0,0])
    pos_neuronas.append(neuronas.copy())
    epoca += 1

fin = time.perf_counter()
print(f"Tiempo transcurrido: {fin - inicio:.4f} segundos")

anim = animation.FuncAnimation(fig=fig, func=update, frames=cant_frames, interval=10, repeat=False)
plt.show()


# Gráfico de centroides con sus patrones

# Generamos las etiquetas para cada patrón y la frecuencia de activacion de cada neurona
etiquetas = []
for entrada in range(datos.shape[0]):
    # Seleccion de neurona ganadora
    neuronas_vector = neuronas.reshape(-1, nro_entradas)
    normas = np.subtract(datos[entrada], neuronas_vector)
    normas = np.linalg.norm(normas, ord=2, axis=1)
    neurona_ganadora = np.argmin(normas)
    etiquetas.append(int(neurona_ganadora))

# Graficamos
fig2, ax2 = plt.subplots()
ax2.scatter(datos[:,0], datos[:,1], c=etiquetas, s=10)
neuronas_vector = neuronas.reshape(-1, nro_entradas)
print(neuronas_vector)
ax2.scatter(neuronas_vector[:,0], neuronas_vector[:,1], c='r', s=60, marker='X', edgecolors='k', label='Centroides')

plt.show()