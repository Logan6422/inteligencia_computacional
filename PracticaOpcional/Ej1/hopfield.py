import numpy as np
import matplotlib.pyplot as plt

patrones = np.array([
    [
        [1,  1,  1,  1, 1],
        [1, -1, -1, -1, 1],
        [1, -1,  1, -1, 1],
        [1, -1, -1, -1, 1],
        [1,  1,  1,  1, 1]
    ],
    [
        [-1, -1,  1,  1,  1],
        [ 1, -1, -1,  1,  1],
        [ 1,  1, -1,  1,  1],
        [ 1,  1, -1, -1,  1],
        [ 1,  1,  1, -1, -1]
    ],
    [
        [-1, -1, -1, -1, -1],
        [-1,  1,  1,  1, -1],
        [-1,  1,  1,  1, -1],
        [-1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1]
    ]
]);

plt.figure(figsize=(6, 3))

for i in range(len(patrones)):
    plt.subplot(1, len(patrones), i + 1);
    plt.imshow(patrones[i], cmap="gray");

    plt.title("Patrón " + str(i + 1));

    plt.xticks(np.arange(-0.5, len(patrones[i][0]), 1));
    plt.yticks(np.arange(-0.5, len(patrones[i]), 1));
    plt.grid(True);

plt.show(block=False);
plt.pause(0.1);

filas = len(patrones[0]);
columnas = len(patrones[0][0]);
patrones = patrones.reshape(len(patrones), filas * columnas);

size = len(patrones[0]);
W = np.zeros((size, size));

#Entrenamiento Hebbiano
for patron in patrones:
    aporte = np.outer(patron, patron);
    W = W + aporte;

np.fill_diagonal(W, 0);
print("Matriz W:");
print(W);

#Recuperación
patron_original = patrones[np.random.randint(len(patrones))];
estado = patron_original.copy();
cantidad_ruido = 2;

posiciones = np.random.choice(size, cantidad_ruido, replace=False);
for posicion in posiciones:
    estado[posicion] *= -1;

resultado = W @ estado;
print("Resultado:");
print(resultado);

#Función de activación
nuevo_estado = np.zeros(size, dtype=int);

for i in range(size):
    if resultado[i] > 0:
        nuevo_estado[i] = 1;
    elif resultado[i] < 0:
        nuevo_estado[i] = -1;
    else:
        nuevo_estado[i] = estado[i];

ruido = estado.reshape(filas, columnas);
recuperado = nuevo_estado.reshape(filas, columnas);

plt.figure(figsize=(6, 3));

plt.subplot(1, 2, 1);
plt.imshow(ruido, cmap="gray");
plt.title("Ruido");
plt.xticks(np.arange(-0.5, columnas, 1));
plt.yticks(np.arange(-0.5, filas, 1));
plt.grid(True);

plt.subplot(1, 2, 2)
plt.imshow(recuperado, cmap="gray");
plt.title("Recuperado");
plt.xticks(np.arange(-0.5, columnas, 1));
plt.yticks(np.arange(-0.5, filas, 1));
plt.grid(True);

plt.show();