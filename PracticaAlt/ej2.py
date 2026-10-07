import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ==========================================
# 1. MEMORIAS FUNDAMENTALES: dígitos 0-9 en grillas 7x5
#    1 = negro, 0 = blanco
# ==========================================
DIGITOS = {
0: [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
1: [[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]],
2: [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]],
3: [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
4: [[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]],
5: [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
6: [[0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
7: [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
8: [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
9: [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[0,1,1,1,0]],
}

def a_patron(g):                      # 1 (negro) -> +1 ; 0 (blanco) -> -1
    return np.where(np.array(g, int).flatten() == 1, 1, -1)

memorias = np.array([a_patron(DIGITOS[d]) for d in range(10)])
P, N = memorias.shape                 # P=10 patrones, N=35 neuronas
print(f"N={N}, P={P}, Pmax teorica={N/(2*np.log(N)):.2f}  -> red sobrecargada")

# ==========================================
# 2. ENTRENAMIENTO HEBBIANO (una sola pasada)
# ==========================================
W = np.zeros((N, N))
for x in memorias:
    W += np.outer(x, x)
W /= N
np.fill_diagonal(W, 0.0)              # w_ii = 0

# ==========================================
# 3. RECUPERACION (dinamica asincrona del PDF)
# ==========================================
def recuperar(W, x, max_sweep=200):
    y = x.copy()
    for _ in range(max_sweep):
        y_ant = y.copy()
        for j in np.random.permutation(N):        # j* = rnd(N)
            h = W[j] @ y
            if h > 0:   y[j] =  1
            elif h < 0: y[j] = -1                 # h == 0 -> mantiene estado
        if np.array_equal(y, y_ant):
            break
    return y

def identificar(y):
    for d in range(P):
        if np.array_equal(y, memorias[d]):
            return d, True                        # convergio a memoria fundamental
    return int(np.argmax(memorias @ y)), False    # estado espurio -> mas cercana

# Diagnostico: que tan puntos fijos son los digitos
for d in range(P):
    ok = np.array_equal(recuperar(W, memorias[d]), memorias[d])
    print(f"Digito {d}: punto fijo = {ok}")

# ==========================================
# 4. INTERFAZ INTERACTIVA (dibujar y recuperar)
# ==========================================
estado = -np.ones((7, 5), int)        # lienzo en blanco

fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(9, 5))
plt.subplots_adjust(bottom=0.22)

def pintar(ax, grid, titulo):
    ax.clear()
    ax.imshow(grid, cmap='gray', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which='minor', color='k', lw=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo, fontsize=10)

pintar(ax_in,  estado, "ENTRADA: click para dibujar / teclas 0-9 cargan digito")
pintar(ax_out, estado, "SALIDA de la red")

def onclick(event):                   # dibujar a mano
    if event.inaxes != ax_in or event.xdata is None: return
    c, r = int(round(event.xdata)), int(round(event.ydata))
    if 0 <= r < 7 and 0 <= c < 5:
        estado[r, c] *= -1
        pintar(ax_in, estado, "ENTRADA: click para dibujar / teclas 0-9 cargan digito")
        fig.canvas.draw_idle()

def on_key(event):                    # atajo: cargar un digito limpio
    if event.key in "0123456789":
        estado[:] = a_patron(DIGITOS[int(event.key)]).reshape(7, 5)
        pintar(ax_in, estado, f"ENTRADA: digito {event.key}")
        fig.canvas.draw_idle()

def btn_recuperar(event):
    y = recuperar(W, estado.flatten())
    d, exacta = identificar(y)
    titulo = f"Recuperado: digito {d}" if exacta else f"Estado espurio (mas cercano: {d})"
    pintar(ax_out, y.reshape(7, 5), titulo)
    fig.canvas.draw_idle()

def btn_limpiar(event):
    estado[:] = -1
    pintar(ax_in,  estado, "ENTRADA: click para dibujar / teclas 0-9 cargan digito")
    pintar(ax_out, estado, "SALIDA de la red")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_key)

b_rec = Button(plt.axes([0.15, 0.06, 0.3, 0.09]), "Recuperar")
b_clr = Button(plt.axes([0.55, 0.06, 0.3, 0.09]), "Limpiar")
b_rec.on_clicked(btn_recuperar)
b_clr.on_clicked(btn_limpiar)

plt.show()