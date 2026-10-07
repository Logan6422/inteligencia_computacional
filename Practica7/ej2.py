import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import random
import time 

datos = np.loadtxt("gr17.csv", delimiter=",", dtype=float)
n = datos.shape[0] 
d = datos
datos_aux = datos.copy()
np.fill_diagonal(datos_aux, np.inf) # por la division por 0
eta = 1.0 / datos_aux
np.fill_diagonal(eta, 0.0)

N_hormigas = n # 17 una por ciudad
alpha = 1.0
beta = 3.0
Q = 100.0 # escala del deposito de feromona
T_max = 200 # iteraciones maximas
sigma0 = 0.1 # feromona inicial

# aristas del recorrido cerrado
def aristas_del_recorrido(recorrido):
    pares = []
    for s in range(n):
        i = recorrido[s]
        j = recorrido[(s + 1) % n]
        pares.append((min(i, j), max(i, j)))
    return sorted(pares) # mismo ciclo -> misma lista ordenada

def AS(rho, metodo, T_max=T_max, seed=0):
    random.seed(seed) # fijo la semilla para que los resultados sean reproducibles
    np.random.seed(seed)

    sigma = np.random.uniform(0, sigma0, (n, n)) # feromona inicial chica y al azar
    # sigma = (sigma + sigma.T) / 2 
    mejor_recorrido = None
    mejor_fpk = np.inf
    t0 = time.perf_counter()
    t = 0
    while t < T_max:
        recorrido = np.empty((N_hormigas, n), dtype=int) # p^k(t) de esta iteracion
        f_pk = np.empty(N_hormigas) # f(p^k(t)) por hormiga
        for k in range(N_hormigas):
            visitadas = np.zeros(n, dtype=bool)
            i = random.randrange(n) # ciudad de partida
            recorrido[k, 0] = i
            visitadas[i] = True
            
            for paso in range(1, n):
                candidatos = np.where(~visitadas)[0] # N^k_i: las no visitadas
                numerador = (sigma[i, candidatos] ** alpha) * (eta[i, candidatos] ** beta)
                p = numerador / numerador.sum()
                j = np.random.choice(candidatos, p=p) # ruleta
                recorrido[k, paso] = j
                visitadas[j] = True
                i = j
            
            f_pk[k] = 0.0
            for paso in range(n - 1):
                f_pk[k] += d[recorrido[k, paso], recorrido[k, paso + 1]]
            f_pk[k] += d[recorrido[k, n - 1], recorrido[k, 0]] # arista de cierre

            if f_pk[k] < mejor_fpk:
                mejor_fpk = f_pk[k]
                mejor_recorrido = recorrido[k].copy()

        # para cada conexion (i, j) evaporacion y deposito
        sigma = (1 - rho) * sigma # sigma_ij(t) = (1−rho)*sigma_ij(t)
        for k in range(N_hormigas):
            # delta que no depende de la arista se calcula una vez por hormiga
            if   metodo == "global":   delta = Q / f_pk[k] # Q/f(p^k(t))
            elif metodo == "uniforme": delta = Q # Q

            for s in range(n): # aristas del recorrido cerrado
                i = recorrido[k, s]
                j = recorrido[k, (s + 1) % n] # el modulo agrega la arista de cierre
                if metodo == "local":
                    delta = Q / d[i, j]  # Q/d_ij depende de la arista
                sigma[i, j] += delta # sigma_ij(t+1) = sigma_ij(t) + sumatoria( delta_sigma^k_ij(t) )
                sigma[j, i] += delta # simetria

        base = aristas_del_recorrido(recorrido[0])
        if all(aristas_del_recorrido(recorrido[k]) == base for k in range(1, N_hormigas)):
            t += 1
            break       
        t += 1
    
    tiempo = time.perf_counter() - t0
    return mejor_recorrido, mejor_fpk, tiempo

R = 10
rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
metodos = ["global", "local", "uniforme"]

tabla = []
for rho in rhos:
    for metodo in metodos:
        Ls, ts = [], []
        for r in range(R):
            rec, L, t_exec = AS(rho, metodo, seed=r)
            Ls.append(L); ts.append(t_exec)
        print(f"rho = {rho:.1f} | {metodo:>8} | media ={np.mean(Ls):7.1f} | desvio ={np.std(Ls):5.1f} | t_medio ={np.mean(ts):5.2f}s")    
        tabla.append((rho, metodo, np.mean(Ls), np.std(Ls), np.mean(ts)))