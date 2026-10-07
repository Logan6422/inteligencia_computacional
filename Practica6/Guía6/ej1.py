import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import clase_core as ev

#minimos globales de las funciones para comparar 
MIN_GLOBAL = {1: -418.9829, 2: 0.0}   # f(420.9687) para i), f(0,0) para ii)
LIM = {1: 512, 2: 100} #limites de los dominios


#LOOP PRINCIPAL
def correr_ag(funcion, n=50, pc=0.8, pm=0.01, max_gen=200, paciencia=30):
    #seteamos las funciones
    gen_pob = ev.Core.generar_poblacion if funcion == 1 else ev.Core.generar_poblacion2
    evaluar = ev.Core.evaluar_poblacion if funcion == 1 else ev.Core.evaluar_poblacion2

    #poblacion inicial aleatoria
    poblacion = gen_pob(n)

    #variables para almacenar parametros
    mejor_hist, prom_hist = [], []
    mejor_x, mejor_val = None, np.inf
    sin_mejora = 0 #paciencia

    for _ in range(max_gen):
        resultados = evaluar(poblacion) #evalua todos los individuos
        fitness = ev.Core.fitness_calc(resultados)
        elite = ev.Core.selec_elite(resultados)  # (cromosoma, x, valor)

        mejor_hist.append(elite[2])
        prom_hist.append(np.mean([r[2] for r in resultados]))

        #si el elite mejora entonces se guarda
        if elite[2] < mejor_val - 1e-9:
            mejor_val, mejor_x, sin_mejora = elite[2], elite[1], 0
        else:
            #se aumenta el contador de paciencia
            sin_mejora += 1
        if sin_mejora >= paciencia:
            break

        #nueva poblacion para proxima gen
        poblacion = ev.Core.generar_poblacion_nueva(poblacion, resultados, fitness, pc, pm)

    return mejor_x, mejor_val, mejor_hist, prom_hist


# DESCENSO POR GRADIENTE 
def correr_gd(funcion, lr, max_iter=2000, tol=1e-6):
    lim = LIM[funcion]
    if funcion == 1:
        x = np.random.uniform(-lim, lim)
        f = ev.Core.funcion_objetivo
        grad = ev.Core.derivada_funcion
    else:
        x = np.random.uniform(-lim, lim, 2)
        f = lambda p: ev.Core.funcion_objetivo2(p[0], p[1])
        grad = lambda p: np.array(ev.Core.derivada_funcion2(p[0], p[1]))

    hist = [f(x)]
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol: #criterio de parada = norma del gradiente 
            break
        x = np.clip(x - lr * g, -lim, lim)               # limitar al dominio
        hist.append(f(x))
    return x, hist[-1], hist


#COMPARACION AG vs GRADIENTE 
def comparar(funcion, lr, n_corridas=30, **ag_params):
    ag_runs, gd_runs, t_ag, t_gd = [], [], [], []
    for _ in range(n_corridas):
        t = time.perf_counter()
        ag_runs.append(correr_ag(funcion, **ag_params))
        t_ag.append(time.perf_counter() - t)

        t = time.perf_counter()
        gd_runs.append(correr_gd(funcion, lr))
        t_gd.append(time.perf_counter() - t)

    err_ag = np.array([r[1] for r in ag_runs]) - MIN_GLOBAL[funcion]
    err_gd = np.array([r[1] for r in gd_runs]) - MIN_GLOBAL[funcion]

    print(f"\n=== Funcion {'i' if funcion == 1 else 'ii'} ({n_corridas} corridas) ===")
    print(f"{'':6s}{'error medio':>14s}{'desvio':>12s}{'mejor':>12s}{'tiempo [s]':>13s}")
    print(f"{'AG':6s}{err_ag.mean():14.4f}{err_ag.std():12.4f}{err_ag.min():12.4f}{np.mean(t_ag):13.4f}")
    print(f"{'GD':6s}{err_gd.mean():14.4f}{err_gd.std():12.4f}{err_gd.min():12.4f}{np.mean(t_gd):13.4f}")
    return ag_runs, gd_runs


# GRAFICOS 
def graficar(funcion, ag_runs, gd_runs):
    ag = min(ag_runs, key=lambda r: r[1])
    gd = min(gd_runs, key=lambda r: r[1])
    fig = plt.figure(figsize=(12, 5))

    if funcion == 1:
        ax1 = fig.add_subplot(1, 2, 1)
        xs = np.linspace(-512, 512, 1024)
        ax1.plot(xs, ev.Core.funcion_objetivo(xs), lw=0.8)
        ax1.scatter(ag[0], ag[1], c='r', s=50, zorder=5, label=f"AG: x={ag[0]:.2f}")
        ax1.scatter(gd[0], gd[1], c='g', s=50, zorder=5, marker='x', label=f"GD: x={float(gd[0]):.2f}")
        ax1.set_title("Funcion i)")
    else:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        xs = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(xs, xs)
        ax1.plot_surface(X, Y, ev.Core.funcion_objetivo2(X, Y), cmap='viridis', alpha=0.6)
        ax1.scatter3D([ag[0][0]], [ag[0][1]], [ag[1]], c='r', s=60, depthshade=False,
                      label=f"AG: ({ag[0][0]:.1f}, {ag[0][1]:.1f})")
        ax1.scatter3D([gd[0][0]], [gd[0][1]], [gd[1]], c='g', s=60, marker='x', depthshade=False,
                      label=f"GD: ({gd[0][0]:.1f}, {gd[0][1]:.1f})")
        ax1.set_title("Funcion ii)")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ag[2], label="AG mejor")
    ax2.plot(ag[3], label="AG promedio", alpha=0.7)
    ax2.plot(gd[2], label="GD", ls='--')
    ax2.set_xlabel("generaciones / iteraciones")
    ax2.set_ylabel("f")
    ax2.set_title("Convergencia")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.show()


#PRUEBAS DE PARAMETROS 
def probar_parametros(funcion, n_corridas=20):
    print(f"\n--- Parametros AG, funcion {'i' if funcion == 1 else 'ii'} ---")
    for n in (20, 50, 100):
        for pc in (0.6, 0.9):
            for pm in (0.001, 0.01, 0.05):
                e = [correr_ag(funcion, n=n, pc=pc, pm=pm)[1] - MIN_GLOBAL[funcion]
                     for _ in range(n_corridas)]
                print(f"n={n:3d} pc={pc} pm={pm}: error medio={np.mean(e):9.4f} desvio={np.std(e):8.4f}")


if __name__ == "__main__":
    for funcion, lr in [(1, 0.5), (2, 0.01)]:
        ag_runs, gd_runs = comparar(funcion, lr, n_corridas=30, n=50, pc=0.8, pm=0.01)
        graficar(funcion, ag_runs, gd_runs)
    # probar_parametros(1)
    # probar_parametros(2)