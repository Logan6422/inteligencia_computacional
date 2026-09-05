import neurona as n
import capa as c
import red as r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Configuracion
# Ejercicio 1 XOR
# datos_trn = pd.read_csv("XOR_trn.csv")
# datos_tst = pd.read_csv("XOR_tst.csv")

# cant_entradas = 2
# # 2 neuronas en capa oculta, 1 en salida
# arquitectura = [2, 1]  
# epocaMax = 50
# porcentaje_corte = 100
# eta = 0.5
# red = r.red(eta, cant_entradas, arquitectura)
# printit = True
# printfinal = False
# graficar = True

# print("ENTRENAMIENTO")
# hist = red.entrenar(printit, printfinal, graficar, datos_trn, epocaMax, porcentaje_corte)

# print("\nTEST")
# aciertos_tst = 0
# for i in range(len(datos_tst)):
#     fila = datos_tst.iloc[i]
#     entrada = np.array(fila.iloc[:cant_entradas], dtype=float)
#     deseada = np.array(fila.iloc[cant_entradas:], dtype=float)
    
#     salida = red.forward_pass(entrada)
#     prediccion = np.where(np.array(salida) >= 0, 1, -1)
    
#     if np.array_equal(prediccion, deseada):
#         aciertos_tst += 1

# porcentaje_tst = (aciertos_tst / len(datos_tst)) * 100
# print(f"Resultados: Aciertos: {aciertos_tst}/{len(datos_tst)} -> Porcentaje: {round(porcentaje_tst, 2)}%")


###############################################################################################################
# Ejercicio 2 Concentricos
# datos_trn = pd.read_csv("concent_trn.csv")
# datos_tst = pd.read_csv("concent_tst.csv")
# cant_entradas = 2
# arquitectura = [2, 8, 1]
# epocaMax = 200
# porcentaje_corte = 100
# eta = 0.5
# red = r.red(eta, cant_entradas, arquitectura)
# printit = True
# printfinal = False # graficar_final es exclusivo de Iris
# graficar = True

# print("ENTRENAMIENTO")
# hist = red.entrenar(printit, printfinal, graficar, datos_trn, epocaMax, porcentaje_corte)

# print("\nTEST")
# aciertos_tst = 0
# for i in range(len(datos_tst)):
#     fila = datos_tst.iloc[i]
#     entrada = np.array(fila.iloc[:cant_entradas], dtype=float)
#     deseada = np.array(fila.iloc[cant_entradas:], dtype=float)

#     salida = red.forward_pass(entrada)
#     prediccion = np.where(np.array(salida) >= 0, 1, -1)

#     if np.array_equal(prediccion, deseada):
#         aciertos_tst += 1

# porcentaje_tst = (aciertos_tst / len(datos_tst)) * 100
# print(f"Resultados: Aciertos: {aciertos_tst}/{len(datos_tst)} -> Porcentaje: {round(porcentaje_tst, 2)}%")
# # red.graficar_zona(datos_trn, "Concéntricos (entrenamiento)")
# # red.graficar_zona(datos_tst, "Concéntricos (test)")

# red.graficar_zona(datos_trn, "Concéntricos (entrenamiento)")
# if hasattr(red, "fig_zona"):
#     plt.close(red.fig_zona); del red.fig_zona;   # fuerza ventana nueva
# red.graficar_zona(datos_tst, "Concéntricos (test)")
# plt.ioff();
# plt.show();

###############################################################################################################
# Ejercicio 3 Iris
datos_trn = pd.read_csv("iris81_trn.csv");
datos_tst = pd.read_csv("iris81_tst.csv");
cant_entradas = 4;
arquitectura = [6,7,3];
epocaMax = 200;
porcentaje_corte = 100;
etas = [0.05, 0.1, 0.25];

resultados = {}

for eta in etas:
    red = r.red(eta, cant_entradas, arquitectura);

    printit = True;
    printfinal = True;
    graficar = False;

    hist = red.entrenar(printit, printfinal, graficar, datos_trn, epocaMax, porcentaje_corte);
    resultados[eta] = hist;

    # test
    aciertos = 0;
    for i in range(len(datos_tst)):
        fila = datos_tst.iloc[i];
        entrada = np.array(fila.iloc[:cant_entradas], dtype=float);
        deseada = np.array(fila.iloc[cant_entradas:], dtype=float);
        salida = np.array(red.forward_pass(entrada), dtype=float);
        if np.argmax(salida) == np.argmax(deseada):
            aciertos += 1;
    print(f"eta={eta} -> TEST: {aciertos}/{len(datos_tst)} ({round(aciertos/len(datos_tst)*100, 2)}%)");

# curvas de error vs epocas para cada tasa
r.graficar_curvas(resultados);