# import numpy as np
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.collections import LineCollection
# import random
# import time

# # Carga de datos
# datos = np.loadtxt("circulo.csv", delimiter=',')

# # Variables
# epocasMax_1 = 200
# epocasMax_2 = 400
# epocasMax_3 = 400
# nro_entradas = datos.shape[1]
# nro_patrones = datos.shape[0]
# tam_matriz = [10,10]
# neuronas = np.empty(shape=(tam_matriz[0],tam_matriz[1],nro_entradas))

# # Matriz de posiciones para medir la distancia (entorno de neurona)
# neuronas_pos = np.indices((tam_matriz[0], tam_matriz[1])).transpose(1,2,0)

# # Inicialización de los pesos (entrada aleatoria)
# indices = np.arange(nro_patrones)
# random.shuffle(indices)
# indice = 0
# for i in range(tam_matriz[0]):
#     for j in range(tam_matriz[1]):
#         neuronas[i,j] = datos[indices[indice]]
#         indice += 1

# # Animación
# fig, ax = plt.subplots()

# pos_neuronas = []

# # Scatter de datos
# scat_datos = ax.scatter(datos[:,0], datos[:,1], s=5)

# # Scatter de neuronas
# x = neuronas[:, :, 0].flatten()
# y = neuronas[:, :, 1].flatten()
# scat_neuronas = ax.scatter(x,y,s=5,c='r')

# # Lineas de neuronas vecinas
# def get_segments(neuronas):
#     segments = []
#     for i in range(tam_matriz[0]):
#         for j in range(tam_matriz[1]):
#             if j < tam_matriz[1] - 1:  # vecino a la derecha
#                 segments.append([neuronas[i, j, :2], neuronas[i, j+1, :2]])
#             if i < tam_matriz[0] - 1:  # vecino de abajo
#                 segments.append([neuronas[i, j, :2], neuronas[i+1, j, :2]])
#     return segments

# lineas = LineCollection(get_segments(neuronas), colors='gray', linewidths=0.5, zorder=1)
# ax.add_collection(lineas)

# # Extras
# texto_frame = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', fontsize=9)
# texto_etapa = ax.text(0.02, 0.88, '', transform=ax.transAxes, ha='left', va='top', fontsize=9)
# cant_frames = epocasMax_1 + epocasMax_2 + epocasMax_3
# ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5))

# def update(frame):
#     x = pos_neuronas[frame][:,:,0].flatten()
#     y = pos_neuronas[frame][:,:,1].flatten()
#     data = np.stack([x,y]).T
#     scat_neuronas.set_offsets(data)

#     lineas.set_segments(get_segments(pos_neuronas[frame]))

#     # Textos
#     texto_frame.set_text(f'Frame {frame+1}/{cant_frames}')
#     etapa_actual = 1
#     if frame<epocasMax_1:
#         etapa_actual = 1
#     elif frame<epocasMax_2:
#         etapa_actual = 2
#     elif frame<epocasMax_3:
#         etapa_actual = 3
#     texto_etapa.set_text(f'Etapa {etapa_actual}')

#     return scat_neuronas

# # Entrenamiento
# # -- Etapa 1
# epoca = 0
# eta = 0.8
# entorno = 3
# print("Etapa 1")
# inicio = time.perf_counter()
# while(epoca<epocasMax_1):
#     print(f"Epoca: {epoca}")

#     for entrada in range(nro_patrones):
#         # neurona_ganadora = [0,0]
#         norma_minima = np.inf

#         # Seleccion de neurona ganadora
#         neuronas_vector = neuronas.reshape(-1, nro_entradas)
#         normas = np.subtract(datos[entrada], neuronas_vector)
#         normas = np.linalg.norm(normas, ord=2, axis=1)
#         pos = np.argmin(normas)
#         neurona_ganadora = [pos//tam_matriz[1],pos%tam_matriz[1]]

#         # for i in range(tam_matriz[0]):
#         #     for j in range(tam_matriz[1]):
#         #         norma = np.linalg.norm((datos[entrada]-neuronas[i,j]), ord=2)
#         #         if (norma<norma_minima):
#         #             norma_minima = norma
#         #             neurona_ganadora = [i,j]

#         # Adaptacion de los pesos
#         distancias = neuronas_pos - neurona_ganadora
#         distancias = np.sum(np.abs(distancias), axis=-1)
#         mascara = distancias <= entorno
#         errores = np.subtract(datos[entrada], neuronas)
#         neuronas[mascara] += eta*errores[mascara]


#         # for i in range(tam_matriz[0]):
#         #     for j in range(tam_matriz[1]):
#         #         dist = np.subtract(neurona_ganadora,[i,j])
#         #         dist = np.sum(abs(dist))
#         #         if (dist<=entorno):
#         #             error = datos[entrada] - neuronas[i,j]
#         #             neuronas[i,j] = neuronas[i,j] + eta*error
#     pos_neuronas.append(neuronas.copy())
#     epoca += 1

# # -- Etapa 2
# epoca = 0
# entorno = np.linspace(entorno, 1, epocasMax_2)
# entorno = np.floor(entorno)
# eta = np.linspace(eta, 0.1, epocasMax_2)
# print("Etapa 2")
# while(epoca<epocasMax_2):
#     print(f"Epoca {epoca}")

#     for entrada in range(nro_patrones):
#         neurona_ganadora = [0,0]
#         norma_minima = np.inf

#         # Selección de neurona ganadora (OPTIMIZAR)
#         neuronas_vector = neuronas.reshape(-1, nro_entradas)
#         normas = np.subtract(datos[entrada], neuronas_vector)
#         normas = np.linalg.norm(normas, ord=2, axis=1)
#         pos = np.argmin(normas)
#         neurona_ganadora = [pos//tam_matriz[0],pos%tam_matriz[1]]

#         # for i in range(tam_matriz[0]):
#         #     for j in range(tam_matriz[1]):
#         #         norma = np.linalg.norm((datos[entrada]-neuronas[i,j]), ord=2)
#         #         if (norma<norma_minima):
#         #             norma_minima = norma
#         #             neurona_ganadora = [i,j]
        
#         # Adaptacion de los pesos
#         distancias = neuronas_pos - neurona_ganadora
#         distancias = np.sum(np.abs(distancias), axis=-1)
#         mascara = distancias <= entorno[epoca]
#         errores = np.subtract(datos[entrada], neuronas)
#         neuronas[mascara] += eta[epoca]*errores[mascara]

#         # for i in range(tam_matriz[0]):
#         #     for j in range(tam_matriz[1]):
#         #         dist = np.subtract(neurona_ganadora,[i,j])
#         #         dist = np.sum(abs(dist))
#         #         if (dist<=entorno[epoca]):
#         #             error = datos[entrada] - neuronas[i,j]
#         #             neuronas[i,j] = neuronas[i,j] + eta[epoca]*error
#     # print(neuronas[0,0])
#     pos_neuronas.append(neuronas.copy())
#     epoca += 1

# # -- Etapa 3
# epoca = 0
# entorno = 0
# eta = 0.01
# print("Etapa 2")
# while(epoca<epocasMax_3):
#     print(f"Epoca: {epoca}")
    
#     for entrada in range(nro_patrones):
#         neurona_ganadora = [0,0]
#         norma_minima = np.inf

#         # Seleccion de neurona ganadora (OPTIMIZAR)
#         neuronas_vector = neuronas.reshape(-1, nro_entradas)
#         normas = np.subtract(datos[entrada], neuronas_vector)
#         normas = np.linalg.norm(normas, ord=2, axis=1)
#         pos = np.argmin(normas)
#         neurona_ganadora = [pos//tam_matriz[0],pos%tam_matriz[1]]

#         # for i in range(tam_matriz[0]):
#         #     for j in range(tam_matriz[1]):
#         #         norma = np.linalg.norm((datos[entrada]-neuronas[i,j]), ord=2)
#         #         if (norma<norma_minima):
#         #             norma_minima = norma
#         #             neurona_ganadora = [i,j]

#         # Adaptacion de los pesos
#         error = datos[entrada] - neuronas[neurona_ganadora[0],neurona_ganadora[1]]
#         neuronas[neurona_ganadora[0],neurona_ganadora[1]] = neuronas[neurona_ganadora[0],neurona_ganadora[1]] + eta*error
#     # print(neuronas[0,0])
#     pos_neuronas.append(neuronas.copy())
#     epoca += 1

# fin = time.perf_counter()
# print(f"Tiempo transcurrido: {fin - inicio:.4f} segundos")

# anim = animation.FuncAnimation(fig=fig, func=update, frames=cant_frames, interval=10, repeat=False)
# plt.show()



import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import random
import time

# ======================= CONFIGURACIÓN =======================
# Para probar con la T, cambiar a: datos = np.loadtxt("te.csv", delimiter=',')
datos = np.loadtxt("circulo.csv", delimiter=',')

# Para probar SOM Unidimensional con la T, cambiar a: tam_matriz = [1, 100]
tam_matriz = [10, 10]

epocasMax_1 = 300
epocasMax_2 = 500
epocasMax_3 = 200

nro_entradas = datos.shape[1]
nro_patrones = datos.shape[0]
neuronas = np.empty(shape=(tam_matriz[0], tam_matriz[1], nro_entradas))

# Inicialización de los pesos (entrada aleatoria)
indices = np.arange(nro_patrones)
random.shuffle(indices)
indice = 0
for i in range(tam_matriz[0]):
    for j in range(tam_matriz[1]):
        # Si hay menos patrones que neuronas, usar módulo para no indexar fuera de rango
        neuronas[i,j] = datos[indices[indice % nro_patrones]]
        indice += 1

# --- OPTIMIZACIONES DE MEMORIA (ESTILO "PUNTEROS" EN NUMPY) ---
# 1. Vista plana: reshape no copia los datos, es un puntero al mismo bloque de memoria.
neuronas_flat = neuronas.reshape(-1, nro_entradas)

# 2. Buffers pre-asignados: Se reservan una sola vez. En el bucle se sobreescriben (in-place).
diff = np.empty_like(neuronas_flat)
errores = np.empty_like(neuronas)

# 3. Grilla de coordenadas para distancia topológica (evita recalcular posiciones en cada iteración)
grid_i, grid_j = np.indices((tam_matriz[0], tam_matriz[1]))

# ======================= ANIMACIÓN =======================
fig, ax = plt.subplots()
pos_neuronas = []

scat_datos = ax.scatter(datos[:,0], datos[:,1], s=5, c='lightgray')
x = neuronas[:, :, 0].flatten()
y = neuronas[:, :, 1].flatten()
scat_neuronas = ax.scatter(x, y, s=10, c='r')

def get_segments(neuronas_grid):
    segments = []
    for i in range(tam_matriz[0]):
        for j in range(tam_matriz[1]):
            if j < tam_matriz[1] - 1:
                segments.append([neuronas_grid[i, j, :2], neuronas_grid[i, j+1, :2]])
            if i < tam_matriz[0] - 1:
                segments.append([neuronas_grid[i, j, :2], neuronas_grid[i+1, j, :2]])
    return segments

lineas = LineCollection(get_segments(neuronas), colors='gray', linewidths=0.5, zorder=1)
ax.add_collection(lineas)

texto_frame = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', fontsize=9)
texto_etapa = ax.text(0.02, 0.88, '', transform=ax.transAxes, ha='left', va='top', fontsize=11)
cant_frames = epocasMax_1 + epocasMax_2 + epocasMax_3
ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5))

def update(frame):
    x = pos_neuronas[frame][:,:,0].flatten()
    y = pos_neuronas[frame][:,:,1].flatten()
    data = np.stack([x,y]).T
    scat_neuronas.set_offsets(data)
    lineas.set_segments(get_segments(pos_neuronas[frame]))
    
    texto_frame.set_text(f'Frame {frame+1}/{cant_frames}')
    limite1 = epocasMax_1
    limite2 = epocasMax_1 + epocasMax_2
    if frame < limite1: etapa_actual = 1
    elif frame < limite2: etapa_actual = 2
    else: etapa_actual = 3
    texto_etapa.set_text(f'Etapa {etapa_actual}')
    return scat_neuronas

# ======================= ENTRENAMIENTO =======================
inicio = time.perf_counter()

# -- Etapa 1
epoca = 0
eta = 0.7
entorno = 3
print("Etapa 1")

while(epoca < epocasMax_1):
    print(f"Epoca: {epoca}")
    orden = np.random.permutation(nro_patrones)
    for entrada in orden:
        # 1. Selección de neurona ganadora (usando buffer pre-asignado 'diff')
        np.subtract(neuronas_flat, datos[entrada], out=diff)
        np.multiply(diff, diff, out=diff) 
        dist_sq = np.sum(diff, axis=1)
        pos = np.argmin(dist_sq)
        i_win, j_win = divmod(pos, tam_matriz[1]) # Corregido: divide por columnas
        
        # 2. Distancia topológica
        dist_topo = np.abs(grid_i - i_win) + np.abs(grid_j - j_win)
        mascara = dist_topo <= entorno
        
        # 3. Adaptación de pesos (In-place / Punteros)
        np.subtract(datos[entrada], neuronas, out=errores)
        mascara_exp = mascara[:, :, np.newaxis] # Vista, no copia memoria
        np.multiply(errores, mascara_exp, out=errores) 
        errores *= eta
        neuronas += errores
        
    pos_neuronas.append(neuronas.copy())
    epoca += 1

# -- Etapa 2
epoca = 0
entorno_arr = np.linspace(entorno, 1, epocasMax_2)
entorno_arr = np.floor(entorno_arr)
eta_arr = np.geomspace(eta, 0.1, epocasMax_2)
print("Etapa 2")

while(epoca < epocasMax_2):
    print(f"Epoca: {epoca}")
    orden = np.random.permutation(nro_patrones)
    eta = eta_arr[epoca]
    entorno = entorno_arr[epoca]
    
    for entrada in orden:
        np.subtract(neuronas_flat, datos[entrada], out=diff)
        np.multiply(diff, diff, out=diff)
        dist_sq = np.sum(diff, axis=1)
        pos = np.argmin(dist_sq)
        i_win, j_win = divmod(pos, tam_matriz[1])
        
        dist_topo = np.abs(grid_i - i_win) + np.abs(grid_j - j_win)
        mascara = dist_topo <= entorno
        
        np.subtract(datos[entrada], neuronas, out=errores)
        mascara_exp = mascara[:, :, np.newaxis]
        np.multiply(errores, mascara_exp, out=errores)
        errores *= eta
        neuronas += errores
        
    pos_neuronas.append(neuronas.copy())
    epoca += 1

# -- Etapa 3
epoca = 0
entorno = 0
eta = 0.01
print("Etapa 3") # Corregido

while(epoca < epocasMax_3):
    print(f"Epoca: {epoca}")
    orden = np.random.permutation(nro_patrones)
    
    for entrada in orden:
        np.subtract(neuronas_flat, datos[entrada], out=diff)
        np.multiply(diff, diff, out=diff)
        dist_sq = np.sum(diff, axis=1)
        pos = np.argmin(dist_sq)
        
        # En la etapa 3, solo se actualiza la ganadora.
        # Usamos una vista (puntero) directo a la neurona ganadora en la memoria.
        vista_ganadora = neuronas_flat[pos]
        
        # Calculamos el error y actualizamos in-place usando diff[0] como buffer temporal
        np.subtract(datos[entrada], vista_ganadora, out=diff[0])
        diff[0] *= eta
        vista_ganadora += diff[0] # Modifica in-place en neuronas_flat y en neuronas
        
    pos_neuronas.append(neuronas.copy())
    epoca += 1

fin = time.perf_counter()
print(f"Tiempo transcurrido: {fin - inicio:.4f} segundos")

# ======================= GRÁFICO FINAL (Requisito del enunciado) =======================
# "grafique los datos de entrenamiento coloreando cada punto según la neurona ganadora 
# correspondiente, y agregue el centroide de cada neurona con otro marcador."

fig_final, ax_final = plt.subplots()

# Calcular neurona ganadora para cada patrón usando broadcasting
diff_final = datos[:, np.newaxis, :] - neuronas_flat[np.newaxis, :, :]
distancias_final = np.sum(diff_final**2, axis=-1)
ganadoras_idx = np.argmin(distancias_final, axis=1)

# Colores para las neuronas (usando un colormap cíclico para que vecinas tengan colores similares)
colors = plt.cm.hsv(np.linspace(0, 1, tam_matriz[0] * tam_matriz[1]))
data_colors = colors[ganadoras_idx]

ax_final.scatter(datos[:,0], datos[:,1], c=data_colors, s=20, alpha=0.7, label='Datos (por ganadora)')
ax_final.scatter(neuronas_flat[:,0], neuronas_flat[:,1], c='black', marker='x', s=40, linewidths=1.5, label='Centroides SOM')

# Dibujar la grilla topológica final
neuronas_grid = neuronas_flat.reshape(tam_matriz[0], tam_matriz[1], -1)
for i in range(tam_matriz[0]):
    for j in range(tam_matriz[1]):
        if j < tam_matriz[1] - 1:
            ax_final.plot([neuronas_grid[i, j, 0], neuronas_grid[i, j+1, 0]],
                    [neuronas_grid[i, j, 1], neuronas_grid[i, j+1, 1]], 'k-', alpha=0.3, linewidth=0.5)
        if i < tam_matriz[0] - 1:
            ax_final.plot([neuronas_grid[i, j, 0], neuronas_grid[i+1, j, 0]],
                    [neuronas_grid[i, j, 1], neuronas_grid[i+1, j, 1]], 'k-', alpha=0.3, linewidth=0.5)
                    
ax_final.legend()
ax_final.set_title("Resultado Final: Mapeo Topológico y Regiones de Voronoi")
ax_final.set_aspect('equal')

# ======================= EJECUCIÓN =======================
anim = animation.FuncAnimation(fig=fig, func=update, frames=cant_frames, interval=10, repeat=False)
plt.show()