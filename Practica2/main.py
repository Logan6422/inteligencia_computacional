import neurona as n
import capa as c
import red as r
import pandas as pd


#Configuracion
datos = pd.read_csv("concent_trn.csv");
cant_entradas = 2;
arquitectura = [8,4,1];
eta = 0.5;
epocaMax = 500;
porcentaje_corte = 99;

red = r.red(eta, cant_entradas, arquitectura);

printit = True;
printfinal = True;
graficar = True;

red.entrenar(printit, printfinal, graficar, datos, epocaMax, porcentaje_corte);
