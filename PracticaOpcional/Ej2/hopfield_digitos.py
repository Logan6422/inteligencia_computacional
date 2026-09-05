import numpy as np

import matplotlib.pyplot as plt

patrones = np.array([

    [#0
        [1,   1,  -1,  -1,  1],
        [1,  -1,   1,   1, -1],
        [1,  -1,   1,   1, -1],
        [1,  -1,   1,   1, -1],
        [1,  -1,   1,   1, -1],
        [1,  -1,   1,   1, -1],
        [1,   1,  -1,  -1,  1]
    ],

    [#1
        [1,   1,  -1,   1, 1],
        [1,  -1,  -1,   1, 1],
        [1,   1,  -1,   1, 1],
        [1,   1,  -1,   1, 1],
        [1,   1,  -1,   1, 1],
        [1,   1,  -1,   1, 1],
        [1,  -1,  -1,  -1, 1]
    ],

    [#2
        [ 1,  -1,  -1,  -1,  1],
        [-1,   1,   1,   1, -1],
        [ 1,   1,   1,   1, -1],
        [ 1,  -1,  -1,  -1,  1],
        [-1,   1,   1,   1,  1],
        [-1,   1,   1,   1,  1],
        [ 1,  -1,  -1,  -1, -1]
    ],
    [#3
        [ 1,  -1,  -1,  -1,  1],
        [-1,   1,   1,   1, -1],
        [ 1,   1,   1,   1, -1],
        [ 1,  -1,  -1,  -1,  1],
        [ 1,   1,   1,   1, -1],
        [-1,   1,   1,   1, -1],
        [ 1,  -1,  -1,  -1,  1]
    ],
    [#4
        [ 1,   1,   1,  -1,  1],
        [ 1,   1,  -1,  -1,  1],
        [ 1,  -1,   1,  -1,  1],
        [-1,   1,   1,  -1,  1],
        [-1,  -1,  -1,  -1, -1],
        [ 1,   1,   1,  -1,  1],
        [ 1,   1,   1,  -1,  1]
    ],
    [#5
        [-1,  -1,  -1, -1, -1],
        [-1,   1,   1,  1,  1],
        [-1,  -1,  -1, -1,  1],
        [ 1,   1,   1,  1, -1],
        [ 1,   1,   1,  1, -1],
        [-1,   1,   1,  1, -1],
        [ 1,  -1,  -1, -1,  1]
    ],
    [#6
        [ 1,   1,  -1,  -1,  1],
        [ 1,  -1,   1,   1,  1],
        [-1,   1,   1,   1,  1],
        [-1,  -1,  -1,  -1,  1],
        [-1,   1,   1,   1, -1],
        [-1,   1,   1,   1, -1],
        [ 1,  -1,  -1,  -1,  1]
    ],
    [#7
        [-1,  -1,  -1,  -1, -1],
        [ 1,   1,   1,   1, -1],
        [ 1,   1,   1,  -1,  1],
        [ 1,   1,  -1,   1,  1],
        [ 1,  -1,   1,   1,  1],
        [ 1,  -1,   1,   1,  1],
        [ 1,  -1,   1,   1,  1]
    ],
    [#8
        [ 1,  -1,  -1,  -1,  1],
        [-1,   1,   1,   1, -1],
        [-1,   1,   1,   1, -1],
        [ 1,  -1,  -1,  -1,  1],
        [-1,   1,   1,   1, -1],
        [-1,   1,   1,   1, -1],
        [ 1,  -1,  -1,  -1,  1]
    ],
    [#9
        [1,  -1,  -1,  -1,  1],
        [-1,  1,   1,   1, -1],
        [-1,  1,   1,   1, -1],
        [1,  -1,  -1,  -1, -1],
        [1,   1,   1,   1, -1],
        [1,   1,   1,  -1,  1],
        [1,  -1,  -1,   1,  1]
    ],

]);

plt.figure(figsize=(8, 5));

for i in range(len(patrones)):
    plt.subplot(2, 5, i + 1);
    plt.imshow(patrones[i], cmap="gray");
    plt.title("Patrón " + str(i));
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

patrones = patrones[:8]; #maximo de patrones para ruido 1
#Entrenamiento Hebbiano
for patron in patrones:
    aporte = np.outer(patron, patron);
    W = W + aporte;

np.fill_diagonal(W, 0);
print("Memorias estables:");

for i, patron in enumerate(patrones):

    resultado = W @ patron;

    nuevo_estado = np.zeros(size, dtype=int);

    for j in range(size):

        if resultado[j] > 0:

            nuevo_estado[j] = 1;

        elif resultado[j] < 0:

            nuevo_estado[j] = -1;

        else:

            nuevo_estado[j] = patron[j];

    print(i, np.array_equal(patron, nuevo_estado));
print("Matriz W:");
print(W);

#Recuperación

patron_original = patrones[np.random.randint(len(patrones))];

estado = patron_original.copy();

cantidad_ruido = 2;

posiciones = np.random.choice(size, cantidad_ruido, replace=False);

for posicion in posiciones:

    estado[posicion] *= -1;

ruido = estado.copy();
print("Original:");

print(patron_original.reshape(filas, columnas));

print("Ruido:");

print(ruido.reshape(filas, columnas));
while True:

    estado_anterior = estado.copy();

    resultado = W @ estado;

    #Función de activación

    nuevo_estado = np.zeros(size, dtype=int);

    for i in range(size):

        if resultado[i] > 0:

            nuevo_estado[i] = 1;

        elif resultado[i] < 0:

            nuevo_estado[i] = -1;

        else:

            nuevo_estado[i] = estado[i];

    estado = nuevo_estado;
    print("Iteracion:");
    print(estado.reshape(filas, columnas));
    if np.array_equal(estado, estado_anterior):

        break;

original = patron_original.reshape(filas, columnas);

ruido = ruido.reshape(filas, columnas);

recuperado = estado.reshape(filas, columnas);

plt.figure(figsize=(9, 3));

plt.subplot(1, 3, 1);

plt.imshow(original, cmap="gray");

plt.title("Original");

plt.xticks(np.arange(-0.5, columnas, 1));

plt.yticks(np.arange(-0.5, filas, 1));

plt.grid(True);

plt.subplot(1, 3, 2);

plt.imshow(ruido, cmap="gray");

plt.title("Ruido");

plt.xticks(np.arange(-0.5, columnas, 1));

plt.yticks(np.arange(-0.5, filas, 1));

plt.grid(True);

plt.subplot(1, 3, 3);

plt.imshow(recuperado, cmap="gray");

plt.title("Recuperado");

plt.xticks(np.arange(-0.5, columnas, 1));

plt.yticks(np.arange(-0.5, filas, 1));

plt.grid(True);

plt.show();