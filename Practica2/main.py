import neurona as n
import capa as c
import red as r
import pandas as pd
import numpy as np


#Configuracion
datos = pd.read_csv("concent_trn.csv");
cant_entradas = 2;
arquitectura = [4,1];
eta = 0.02;
epocaMax = 1000;
porcentaje_corte = 90;

red = r.red(eta, cant_entradas, arquitectura);

printit = True;
printfinal = True;
graficar = False;

red.entrenar(printit, printfinal, graficar, datos, epocaMax, porcentaje_corte);

#//////////////////////////////////////////////////////////////////////////////

datos_tst = pd.read_csv("concent_tst.csv")
print("\nTEST")
aciertos_tst = 0
for i in range(len(datos_tst)):
    fila = datos_tst.iloc[i]
    entrada = np.array(fila.iloc[:cant_entradas], dtype=float)
    deseada = np.array(fila.iloc[cant_entradas:], dtype=float)

    salida = red.forward_pass(entrada)
    prediccion = np.where(np.array(salida) >= 0, 1, -1)

    if prediccion == deseada:
        aciertos_tst += 1

        # print(aciertos_tst)
        # print('//////////////')


