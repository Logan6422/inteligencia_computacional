import clase_core as ev

pc = 0.8; #probabilidad de crossover
pm = 0.01; #probabilidad de mutacion
nro_poblacion = 50;
cant_generaciones = 1000;

#PRIMERA ETAPA (INICIALIZAR POBLACION)
#cromosoma de 10 bits (2^10 = 1024 -> [-512, 512])
poblacion = ev.Core.generar_poblacion(nro_poblacion);

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