import neurona as n
import capa as c
import red as r
import pandas as pd


#Configuracion
datos = pd.read_csv("iris81_trn.csv");
cant_entradas = 4;
arquitectura = [6,7,3];
eta = 0.25;
epocaMax = 500;
porcentaje_corte = 90;

red = r.red(eta, cant_entradas, arquitectura);

printit = True;
printfinal = True;
graficar = False;

red.entrenar(printit, printfinal, graficar, datos, epocaMax, porcentaje_corte);
