import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import clase_core as ev
import random
import time

cant_particulas = 4
c1 = 1.5
c2 = 1.5
paciencia = 50

# Inicializacion
D = 1 # funcion i D = 1, funcion ii D = 2
if D == 1:
    xmin = np.full(D, -512.0) # dominio de la funcion i)
    xmax = np.full(D,  512.0)
else:
    xmin = np.full(D, -100.0) # dominio de la funcion ii)
    xmax = np.full(D,  100.0)

vmax = (xmax - xmin)/2
x = np.empty((cant_particulas, D))
v = np.empty((cant_particulas, D))
y = np.empty((cant_particulas, D))
f_yk = np.empty(cant_particulas)
mejor_valor = np.inf
y_global = None
for k in range(cant_particulas):
    for i in range(D):
        x[k, i] = random.uniform(xmin[i], xmax[i]) # x_ki(0) -> U(xmin, xmax)
        v[k, i] = random.uniform(-vmax[i], vmax[i]) # v_ki(0) -> U(-vmax, vmax)

    y[k] = x[k].copy() # y_k = x_k(0)
    f_yk[k] = ev.Core.funcion_objetivo(x[k, 0]) if D == 1 else ev.Core.funcion_objetivo2(x[k, 0], x[k, 1]) # formula i) o ii)
    if f_yk[k] < mejor_valor:
        mejor_valor = f_yk[k]
        y_global = x[k].copy()

# historico de posiciones para la animacion
historico_pos = []
historico_mejor_global = []
historico_pos.append(x.copy())
historico_mejor_global.append(y_global.copy())

# Loop
cant_iteraciones = 1000
historial = []
sin_mejoras = 0
it = 0
while it < cant_iteraciones and sin_mejoras < paciencia:
    previo = mejor_valor

    # Evaluar y actualizar mejores
    for k in range(cant_particulas):
  
        f_xk = ev.Core.funcion_objetivo(x[k, 0]) if D == 1 else ev.Core.funcion_objetivo2(x[k, 0], x[k, 1]) # formula i) o ii)

        if (f_xk < f_yk[k]):
            y[k] = x[k].copy()
            f_yk[k] = f_xk
        if (f_yk[k] < mejor_valor):
            mejor_valor = f_yk[k]
            y_global = y[k].copy()

    historial.append(mejor_valor)
    sin_mejoras = 0 if mejor_valor < previo else sin_mejoras + 1

    # Mover enjambre
    for k in range(cant_particulas):
        for i in range(D):
            r1 = random.random()
            r2 = random.random()
            v[k, i] = v[k, i] + c1 * r1 * (y[k, i] - x[k, i]) + c2 * r2 * (y_global[i] - x[k, i])
            # Limitar velocidad
            if v[k, i] > vmax[i]: v[k, i] = vmax[i]
            if v[k, i] < -vmax[i]: v[k, i] = -vmax[i]

            x[k, i] = x[k, i] + v[k, i]
            # Limitar posicion al dominio
            if x[k, i] > xmax[i]:
                x[k, i] = xmax[i]
                v[k, i] = 0.0
            elif x[k, i] < xmin[i]:
                x[k, i] = xmin[i]
                v[k, i] = 0.0

    # enjambre despues de la iteracion
    historico_pos.append(x.copy())
    historico_mejor_global.append(y_global.copy())

    it += 1

print("Mejor posicion (y_global):", y_global)
print("Mejor error f(y_global):", mejor_valor)


# animacion del enjambre
if D == 1:
    x1a = np.linspace(xmin[0], xmax[0], 1024)
    f1a = np.array([ev.Core.funcion_objetivo(val) for val in x1a])
    fig_anim, ax_anim = plt.subplots()
    ax_anim.plot(x1a, f1a, lw=0.8)
    scat_part = ax_anim.scatter([], [], c='r', s=20)
    scat_best = ax_anim.scatter([], [], c='g', s=60)
    ax_anim.set_xlim(xmin[0], xmax[0])
    ax_anim.set_ylim(f1a.min() - 10, f1a.max() + 10)
    ax_anim.set_title("Enjambre buscando el minimo (funcion i)")
    ax_anim.grid()

    def update(frame):
        pos = historico_pos[frame]
        fpos = np.array([ev.Core.funcion_objetivo(val) for val in pos[:, 0]])
        scat_part.set_offsets(np.stack([pos[:, 0], fpos]).T)
        b = historico_mejor_global[frame]
        scat_best.set_offsets([[b[0], ev.Core.funcion_objetivo(b[0])]])
        return scat_part, scat_best
else:
    x2a = np.linspace(xmin[0], xmax[0], 100)
    y2a = np.linspace(xmin[1], xmax[1], 100)
    Xa, Ya = np.meshgrid(x2a, y2a)
    Za = ev.Core.funcion_objetivo2(Xa, Ya)
    fig_anim = plt.figure(figsize=(7, 7))
    ax_anim = fig_anim.add_subplot(projection='3d')
    ax_anim.plot_surface(Xa, Ya, Za, cmap='viridis', alpha=0.6)
    scat_part = ax_anim.scatter3D([], [], [], c='r', s=20)
    scat_best = ax_anim.scatter3D([], [], [], c='g', s=80, depthshade=False)
    ax_anim.set_xlabel('x'); ax_anim.set_ylabel('y'); ax_anim.set_zlabel('f(x,y)')
    ax_anim.set_title("Enjambre buscando el minimo (funcion ii)")

    def update(frame):
        pos = historico_pos[frame]
        zp = ev.Core.funcion_objetivo2(pos[:, 0], pos[:, 1])
        scat_part._offsets3d = (pos[:, 0], pos[:, 1], zp)
        b = historico_mejor_global[frame]
        zb = ev.Core.funcion_objetivo2(b[0], b[1])
        scat_best._offsets3d = ([b[0]], [b[1]], [zb])
        return scat_part, scat_best

anim = animation.FuncAnimation(fig=fig_anim, func=update, frames=len(historico_pos), interval=50, repeat=False)
plt.show()

if D == 1:
    # Grafico formula i)
    x1 = np.linspace(-512,512,1024)
    f1 = ev.Core.funcion_objetivo(x1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(x1, f1, lw=0.8)
    ax[0].scatter(y_global[0], mejor_valor, c='r', s=40, zorder=5)
    ax[0].set_title(f"Funcion i): minimo en x={y_global[0]:.2f}")
    ax[0].grid()
    ax[1].plot(historial)
    ax[1].set_xlabel("iteraciones")
    ax[1].set_ylabel("f(y_global)")
    ax[1].set_title("Convergencia")
    ax[1].grid()
    plt.tight_layout()
    plt.show()
else:
    # Grafico formula ii)
    x2 = np.linspace(-100, 100, 100)
    y2 = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x2, y2)
    Z = ev.Core.funcion_objetivo2(X, Y)
    fig = plt.figure(figsize=(11, 6))

    # superficie 3D 
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
    ax1.scatter3D([y_global[0]], [y_global[1]], [mejor_valor], s=60, c='r', depthshade=False)
    ax1.set_title(f"Funcion ii): minimo en ({y_global[0]:.2f}, {y_global[1]:.2f})")
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('f(x,y)')
    # convergencia
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(historial)
    ax2.set_xlabel("iteraciones")
    ax2.set_ylabel("f(y_global)")
    ax2.set_title("Convergencia")
    ax2.grid()
    plt.tight_layout()
    plt.show()

