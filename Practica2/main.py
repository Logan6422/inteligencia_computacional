import neurona as n
import capa as c
import red as r
import pandas as pd

#Configuracion
datos = pd.read_csv("iris81_trn.csv");
cant_entradas = 4;
arquitectura = [6,7,3];
epocaMax = 500;
porcentaje_corte = 67; # Para Iris conviene un corte alto para ver bien la convergencia
etas = [0.05, 0.1, 0.25]; 

resultados = {}

for eta in etas:
    red = r.red(eta, cant_entradas, arquitectura);

    # Desactivamos los prints por época y el gráfico final para que corra rápido y limpio
    printit = True; 
    printfinal = True;
    graficar = False;

    # Entrenamos y guardamos el historial que devuelve el método
    hist = red.entrenar(printit, printfinal, graficar, datos, epocaMax, porcentaje_corte);
    resultados[eta] = hist


