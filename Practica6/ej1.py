import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import clase_core as ev
import random
import time

x = np.linspace(-512,512,1024)

y1 = -x*np.sin(np.sqrt(abs(x)))

fig, ax = plt.subplots()

ax.plot(x,y1)
ax.grid()

# plt.show()


pc = 0.8; #probabilidad de crossover
pm = 0.01; #probabilidad de mutacion
nro_poblacion = 50;
cant_generaciones = 1000;
paciencia = 50

#PRIMERA ETAPA (INICIALIZAR POBLACION)
#cromosoma de 10 bits (2^10 = 1024 -> [-512, 512])
poblacion = ev.Core.generar_poblacion(nro_poblacion);

contador_paciencia = 0
mejor_absoluto = 0
for generacion in range(cant_generaciones):
    #SEGUNDA ETAPA (EVALUAR POBLACION)
    resultados = ev.Core.evaluar_poblacion(poblacion);
    # for resultado in resultados:
    #     print(resultado);

    mejor = ev.Core.selec_elite(resultados);
    print(
        "Generacion:", generacion,
        "| x:", mejor[1],
        "| f(x):", mejor[2]
    );

    # if (mejor[2] > mejor_absoluto):


    #TERCERA ETAPA (CALCULAR FITNESS)
    fitness = ev.Core.fitness_calc(resultados);
    # print("Fitness:", fitness);

    poblacion = ev.Core.generar_poblacion_nueva(
        poblacion,resultados,fitness,pc,pm
    );


#RESULTADO FINAL
resultados = ev.Core.evaluar_poblacion(poblacion);
mejor = ev.Core.selec_elite(resultados);

print("\nMEJOR RESULTADO FINAL:");
print("Cromosoma:", mejor[0]);
print("x:", mejor[1]);
print("f(x):", mejor[2]);


# Método del gradiente descendiente
print("Metodo del gradiente:")
x = random.randint(-512,512)
print(f'Valor inicial: {x}')
eta = 0.1
for generacion in range(cant_generaciones):
    x = x-eta*ev.Core.derivada_funcion(x)
    if (x < -512):
        x = -512
    if (x > 512):
        x = 512

print(f'Minimo obtenido: {x}')
y = ev.Core.funcion_objetivo(x)
print(y)

ax.scatter(x,y,s=20,c='r')

# plt.show()


# Función 2:
#PRIMERA ETAPA (INICIALIZAR POBLACION)
#cromosoma de 16 bits (8 por coordenada)
poblacion2 = ev.Core.generar_poblacion2(nro_poblacion);

contador_paciencia = 0
mejor_absoluto = 0
for generacion in range(cant_generaciones):
    #SEGUNDA ETAPA (EVALUAR POBLACION)
    resultados = ev.Core.evaluar_poblacion2(poblacion);
    # for resultado in resultados:
    #     print(resultado);

    mejor = ev.Core.selec_elite(resultados);
    xy = mejor
    print(
        "Generacion:", generacion,
        "| x:", x,
        "| y:", y,
        "| f(x,y):", mejor[2]
    );


    #TERCERA ETAPA (CALCULAR FITNESS)
    fitness = ev.Core.fitness_calc(resultados);
    # print("Fitness:", fitness);

    poblacion = ev.Core.generar_poblacion_nueva(
        poblacion,resultados,fitness,pc,pm
    );


#RESULTADO FINAL
resultados = ev.Core.evaluar_poblacion2(poblacion);
mejor = ev.Core.selec_elite(resultados);

print("\nMEJOR RESULTADO FINAL:");
print("Cromosoma:", mejor[0]);
(x,y) = mejor[1]
print("x:", x);
print("y:", y);
print("f(x,y):", mejor[2]);

x2 = np.linspace(-100,100,100)
y2 = np.linspace(-100,100,100)

X, Y = np.meshgrid(x2,y2)
Z = ev.Core.funcion_objetivo2(X,Y)

fig2, ax2 = plt.subplots()

ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X,Y,Z,cmap='viridis')

ax2.scatter3D(x,y,mejor[2],s=50,c='r')

plt.show()
