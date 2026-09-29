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

    def binario_a_entero(cromosoma):
        binario = ''.join(map(str,cromosoma)); #pasa el array a string
        return int(binario, 2); #pasa el string a int

    def decodificar(entero):
        x = -512 + (entero/1023) * (512 - (-512));
        return x;

    def funcion_objetivo(x):
        return -x * np.sin(np.sqrt(abs(x)));

    def evaluar_individuo(cromosoma):
        v = Core.binario_a_entero(cromosoma);
        x = Core.decodificar(v);
        valor = Core.funcion_objetivo(x);
        return x, valor;

    def evaluar_poblacion(poblacion):
        resultados = [];
        for cromosoma in poblacion:
            x, valor = Core.evaluar_individuo(cromosoma);
            resultados.append((cromosoma,x, valor));

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