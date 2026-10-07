import numpy as np
import matplotlib.pyplot as plt


# Memoria 1: Letra "T"
p1 = np.array([
    [-1,  -1,   -1,  -1,  -1],
    [-1, 1,  1, 1,  -1],
    [-1, 1,  -1, 1,  -1],
    [-1, 1,  1, 1,  -1],
    [-1,  -1,   -1,  -1,  -1]
]).flatten()

# Memoria 2: Letra "L"
p2 = np.array([
    [  1, 1, -1,-1,-1],
    [ -1, 1, 1,-1,-1],
    [ -1,-1, 1, -1,-1],
    [ -1, -1, 1, 1,-1],
    [ -1,-1,-1, 1, 1]
]).flatten()

# Memoria 3: Letra "U"
p3 = np.array([
    [ 1, 1, 1, 1, 1],
    [ 1, -1, -1, -1, 1],
    [ 1, -1, -1, -1, 1],
    [ 1, -1, -1, -1, 1],
    [ 1, 1, 1, 1, 1]
]).flatten()

memorias = [p1, p2, p3]
N = len(p1) # Cantidad de neuronas (25)


# 2. ENTRENAMIENTO (ALMACENAMIENTO)
def entrenar_hopfield(memorias, N):
    """Aprendizaje Hebbiano según las diapositivas."""
    W = np.zeros((N, N))
    for p in memorias:
        W += np.outer(p, p) # Producto externo x_k * x_k^T
    
    W = W / N               # Normalización por N
    # Condición teórica: w_ii = 0 para todo i (sin auto-conexión)
    np.fill_diagonal(W, 0)  
    return W

W = entrenar_hopfield(memorias, N)

# 3. FUNCIÓN DE RUIDO
def agregar_ruido(x, prob_inversion=0.25):
    #Invierte píxeles con una probabilidad dada (25% de ruido)
    x_ruidoso = x.copy()
    mascara = np.random.rand(N) < prob_inversion
    x_ruidoso[mascara] = -x_ruidoso[mascara]
    return x_ruidoso

# 4. PRUEBA (RECUPERACIÓN DINÁMICA)
def recuperar(W, x_ruidoso, max_iter=100):
   #Iteración asíncrona (j* = rnd(N)) hasta convergencia
    y = x_ruidoso.copy()
    
    for _ in range(max_iter):
        y_ant = y.copy()
        indices = np.random.permutation(N)
        
        for j in indices:
            h = np.dot(W[j, :], y) # Campo local
            if h > 0:
                y[j] = 1
            elif h < 0:
                y[j] = -1
            # Si h == 0, la neurona mantiene su estado (sgn(0) = y(n-1))
            
        # Condición de corte: hasta no observar cambios en las y_j
        if np.array_equal(y, y_ant):
            break
            
    return y

# 5. VISUALIZACIÓN Y EJECUCIÓN
def plot_patron(ax, patron, titulo):
    ax.imshow(patron.reshape(5,5), cmap='gray_r', vmin=-1, vmax=1)
    ax.set_title(titulo, fontsize=10)
    ax.axis('off')

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle("Ejercicio 1", fontsize=14)

nombres = ["Memoria 1", "Memoria 2", "Memoria 3"]

for i, p in enumerate(memorias):
    # 1. Patrón Original
    plot_patron(axes[i, 0], p, f"{nombres[i]}\n(Original)")
    
    # 2. Patrón Ruidoso (Entrada a la red)
    p_ruidoso = agregar_ruido(p, prob_inversion=0.30) # 30% de ruido
    plot_patron(axes[i, 1], p_ruidoso, "Entrada Ruidosa\n(y(0) = x)")
    
    # 3. Patrón Recuperado (Salida de la red)
    p_recuperado = recuperar(W, p_ruidoso)
    plot_patron(axes[i, 2], p_recuperado, "Recuperado\n(y(M))")

plt.tight_layout()
plt.show()