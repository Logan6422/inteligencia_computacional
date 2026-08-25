import neurona as n
import capa as c
import red as r
import pandas as pd


#Configuracion
datos = pd.read_csv("XOR_trn.csv");
cant_entradas = 2;
arquitectura = [4, 3, 2, 1];
eta = 0.5;
epocaMax = 500;
porcentaje_corte = 95;

red = r.red(eta, cant_entradas, arquitectura);

red.entrenar(datos, epocaMax, porcentaje_corte);
