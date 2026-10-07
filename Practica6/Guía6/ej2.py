import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import clase_core as ev

# Parametros 
N_POB = 50
PC = 0.8
GENERACIONES = 100
PACIENCIA = 20
LAMBDA = 0.1          # peso de la penalizacion por cantidad de caracteristicas
P_INICIAL = 0.01      # prob. de que un bit empiece en 1 (subconjuntos chicos)
K_VECINOS = 3
SEED = 0
random.seed(SEED)
np.random.seed(SEED)


# Cargar CSV 
def cargar(path):
    datos = np.loadtxt(path, delimiter=",")      
    return datos[:, :-1], datos[:, -1]           
Xtr, ytr = cargar("leukemia_train.csv")
Xte, yte = cargar("leukemia_test.csv")
L = Xtr.shape[1]                                 # 7129
PM = 2.0 / L                                     # ~2 bits mutados por hijo
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

#escalamos los valores y los usamos en el knn
def modelo():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K_VECINOS))

# Cromosoma de L bits 
def generar_poblacion_leucemia(n):
    return [[1 if random.random() < P_INICIAL else 0 for _ in range(L)] for _ in range(n)]


#Fitness (se MAXIMIZA)
cache = {} #memoria para ahorrar evaluaciones caras de cromosoma
def fitness(cromosoma):
    clave = bytes(cromosoma)
    if clave in cache:
        return cache[clave]
    idx = np.flatnonzero(cromosoma) #obtiene indices donde estan los 1
    if len(idx) == 0:                            #caso = 0 caracteristicas
        f = 1e-9
    else:
        acc = cross_val_score(modelo(), Xtr[:, idx], ytr, cv=cv).mean() #calcula el accuracy usando el modelo knn y calcula la media
        f = max(acc - LAMBDA * len(idx) / L, 0.0) + 1e-6   # > 0 para la ruleta
        #fitness = acurracy - penalizacion
        #penaizacion = lambda * genes seleccionados / 71291
    cache[clave] = f #guarda en cache [cromosoma] = accuracy
    return f


def nueva_poblacion(poblacion, fits):
    # elitismo + ruleta/crossover/mutacion de clase_core
    nueva = [poblacion[int(np.argmax(fits))].copy()]
    while len(nueva) < len(poblacion):
        p1, p2 = ev.Core.selec_padres(poblacion, fits, 2)
        h1, h2 = ev.Core.crossover(p1, p2, PC)
        nueva.append(ev.Core.mutacion(h1, PM))
        if len(nueva) < len(poblacion):
            nueva.append(ev.Core.mutacion(h2, PM))
    return nueva


#Loop del AG 
poblacion = generar_poblacion_leucemia(N_POB)
mejor_hist, prom_hist = [], []
mejor_f, mejor_crom, sin_mejora = -np.inf, None, 0
for g in range(GENERACIONES):
    fits = [fitness(c) for c in poblacion] #fitnes para cada cromosoma de la poblacion
    i = int(np.argmax(fits)) #mejor actual
    mejor_hist.append(fits[i])
    prom_hist.append(np.mean(fits))
    if fits[i] > mejor_f + 1e-9:
        mejor_f, mejor_crom, sin_mejora = fits[i], poblacion[i].copy(), 0
    else:
        sin_mejora += 1
    print(f"gen {g:3d} | mejor={fits[i]:.4f} | prom={np.mean(fits):.4f} | n_carac={sum(poblacion[i])}")
    if sin_mejora >= PACIENCIA:
        break
    poblacion = nueva_poblacion(poblacion, fits)


#Evaluacion en TEST
idx = np.flatnonzero(mejor_crom)
acc_sub = modelo().fit(Xtr[:, idx], ytr).score(Xte[:, idx], yte)
acc_all = modelo().fit(Xtr, ytr).score(Xte, yte)
print(f"\nTodas las caracteristicas ({L}): accuracy test = {acc_all:.3f}")
print(f"Subconjunto del AG ({len(idx)}):  accuracy test = {acc_sub:.3f}")


# ---------- 6) Graficos ----------

n_carac_ag = len(idx)
reduccion = 100 * (1 - n_carac_ag / L)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))


# 1) Evolución del fitness
ax[0].plot(
    mejor_hist,
    label="Mejor fitness",
    linewidth=2
)

ax[0].plot(
    prom_hist,
    label="Fitness promedio",
    linewidth=2,
    alpha=0.7
)

ax[0].set_title("Evolución del algoritmo genético")
ax[0].set_xlabel("Generación")
ax[0].set_ylabel("Fitness")
ax[0].legend()
ax[0].grid(alpha=0.3)

# 2) Cantidad de características

ax[1].bar(
    ["Todas", "Subconjunto AG"],
    [L, n_carac_ag]
)

ax[1].set_title("Reducción de características")
ax[1].set_ylabel("Cantidad de características")

ax[1].text(
    0,
    L,
    f"{L}",
    ha="center",
    va="bottom",
    fontweight="bold"
)

ax[1].text(
    1,
    n_carac_ag,
    f"{n_carac_ag}",
    ha="center",
    va="bottom",
    fontweight="bold"
)

ax[1].text(
    1,
    L * 0.5,
    f"Reducción: {reduccion:.1f}%",
    ha="center",
    va="center",
    fontsize=11
)

ax[1].grid(axis="y", alpha=0.3)


# 3) Accuracy sobre TEST
ax[2].bar(
    ["Todas", "Subconjunto AG"],
    [acc_all, acc_sub]
)

ax[2].set_title("Accuracy sobre conjunto de TEST")
ax[2].set_ylabel("Accuracy")
ax[2].set_ylim(0, 1.05)

ax[2].text(
    0,
    acc_all,
    f"{acc_all:.3f}",
    ha="center",
    va="bottom",
    fontweight="bold"
)

ax[2].text(
    1,
    acc_sub,
    f"{acc_sub:.3f}",
    ha="center",
    va="bottom",
    fontweight="bold"
)

ax[2].grid(axis="y", alpha=0.3)


plt.suptitle(
    "Selección de características mediante Algoritmo Genético",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()