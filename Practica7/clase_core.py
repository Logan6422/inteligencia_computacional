import random
import numpy as np

class Core:
    #10 bits por cromosoma
    def generar_poblacion(n):
        poblacion = []
        for i in range(n):
            cromosoma = [random.randint(0,1) for _ in range(10)];
            poblacion.append(cromosoma);

        return poblacion

    def generar_poblacion2(n):
        poblacion = []
        for i in range(n):
            cromosoma = [random.randint(0,1) for _ in range(16)]
            poblacion.append(cromosoma)
        return poblacion

    def binario_a_entero(cromosoma):
        binario = ''.join(map(str,cromosoma)); #pasa el array a string
        return int(binario, 2); #pasa el string a int

        # potencias = []
        # for i in range(10,0,-1):
        #     potencias.append(2^i)
        # return np.dot(potencias,cromosoma)

    def decodificar(entero):
        return -512 + (entero/1023) * (512 - (-512));

    def decodificar2(entero):
        return -100 + (entero/255) * 200;

    def funcion_objetivo(x):
        return -x * np.sin(np.sqrt(abs(x)));

    def derivada_funcion(x):
        return -np.sin(np.sqrt(abs(x)))-(np.sqrt(abs(x))*np.cos(np.sqrt(abs(x))))/2

    def funcion_objetivo2(x,y):
        return (x**2 + y**2)**(0.25) * (np.sin(50*(x**2 + y**2)**(0.1))**2 + 1)

    def derivada_funcion2(x,y):
        s = x**2 + y**2 + 1e-12          # evita división por 0 en el origen
        u = 50 * s**0.1
        dfds = 0.25 * s**-0.75 * (np.sin(u)**2 + 1) + 5 * s**-0.65 * np.sin(2*u)
        return 2*x*dfds, 2*y*dfds

    def evaluar_individuo(cromosoma):
        v = Core.binario_a_entero(cromosoma);
        x = Core.decodificar(v);
        valor = Core.funcion_objetivo(x);
        return x, valor;

    def evaluar_individuo2(cromosoma):
        x = cromosoma[:8]   # los primeros 8 bits
        y = cromosoma[8:]   # los ultimos 8 bits
        vx = Core.binario_a_entero(x)
        vy = Core.binario_a_entero(y)
        vx = Core.decodificar2(vx)
        vy = Core.decodificar2(vy)
        valor = Core.funcion_objetivo2(vx,vy)
        return (vx,vy), valor

    def evaluar_poblacion(poblacion):
        resultados = [];
        for cromosoma in poblacion:
            x, valor = Core.evaluar_individuo(cromosoma);
            resultados.append((cromosoma,x, valor));

        return resultados

    def evaluar_poblacion2(poblacion):
        resultados = [];
        for cromosoma in poblacion:
            xy, valor = Core.evaluar_individuo2(cromosoma);
            resultados.append((cromosoma, xy, valor));

        return resultados

    def fitness_calc(resultados):
        fmax = max(valor for cromosoma, x, valor in resultados);
        fitness = [];
        for cromosoma, x, valor in resultados:
            v = fmax - valor + 1; #(1 garantiza peor fitness = 1)
            fitness.append(v);
        
        return fitness;

    def selec_elite(resultados):
        #funcion lambda anonima
        mejor = min(resultados, key=lambda resultado: resultado[2]);
        return mejor;

    def selec_padres(poblacion, fitness, cantidad):
        padres = random.choices(poblacion, weights=fitness, k=cantidad);
        return padres;

    def crossover(padre1, padre2, pc): #pc = probabilidade de crossover
        if random.random() < pc:
            punto = random.randint(1, len(padre1) - 1);
            hijo1 = padre1[:punto] + padre2[punto:];
            hijo2 = padre2[:punto] + padre1[punto:];
            return hijo1, hijo2

        return padre1.copy(), padre2.copy(); #si no se cruzan siguen los padres como hijos


    def mutacion(cromosoma, pm): #prob mutar
        cromosoma = cromosoma.copy();
        for i in range(len(cromosoma)):
            if random.random() < pm:
                cromosoma[i] = 1 - cromosoma[i]; #mutacion aleatoria
        
        return cromosoma;

    def generar_poblacion_nueva(poblacion, resultados, fitness, pc, pm):
        nueva_poblacion = [];
        elite = Core.selec_elite(resultados);
        nueva_poblacion.append(elite[0]); #extrae el cromosoma elite

        while len(nueva_poblacion) < len(poblacion):
            padres = Core.selec_padres(poblacion, fitness, 2);
            padre1 = padres[0];
            padre2 = padres[1];

            hijo1, hijo2 = Core.crossover(padre1, padre2, pc);
            hijo1 = Core.mutacion(hijo1, pm);
            hijo2 = Core.mutacion(hijo2, pm);

            nueva_poblacion.append(hijo1);

            if len(nueva_poblacion) < len(poblacion):
                nueva_poblacion.append(hijo2);

        return nueva_poblacion;