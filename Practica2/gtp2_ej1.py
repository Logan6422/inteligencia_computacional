import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#b = 1
def sigmoidea(z):
    return (2/(1 + np.exp(-z)))-1

def derivada_sigmoidea(y):
    return 0.5*(1 + y)*(1-y)

class neurona:
    pesos = [],
    pesobias = 0

    def __init__(self,cant_in):
        self.pesobias = np.random(-0.5,0.5,1);
        self.pesos = np.random(-0.5,0.5,cant_in);

    def prod_pto(self,input):
        res.append(self.pesobias);
        res = np.dot(pesos, input);
        return res

    def calc_delta(self, deseada, output):
        error = (deseada - output)
        deriv = derivada_sigmoidea(output)
        delta_actual = error*deriv
        return delta_actual

    def calc_delta(self,output, delta_prev, pesos_prev):
        deriv = derivada_sigmoidea(output)
        delta_actual = delta_prev * pesos_prev * deriv
        return delta_actual


    
        


class capa:
    neuronas = []
    output = []
    delta = []

    def backward_pass(self,deseada):
        #chequear numpy.resta
        #aplicar delta
        self.delta.append(self.neuronas.calc_delta(deseada, self.output))

    def backward_pass(self, delta_prev, pesos_prev):
        for i in range(self.output):
            self.delta.append(self.neuronas.calc_delta(delta_prev,self.output, delta_prev, pesos_prev))

    def forward_pass(self,input):
        input.append(-1)
        for i in range(self.neuronas):
            prod_neu = self.neuronas[i].prod_pto(input)
            self.output.append(sigmoidea(prod_neu));

    def __init__(self, neuronas):
        self.neuronas = neuronas,
        self.output = []

class red:
    capas = [];
    cantInput = 0;
    datos = pd.DataFrame
    #definir datos de panda


    def __init__(self,cantInput):
        self.capas = [],
        self.cantInput = cantInput
        self.datos = pd.DataFrame
    #chequear uso de len
    def init_red(self,cantNeuPorCapas):
        for i in range (cantNeuPorcapas):
            arrayNeuronas = []
            for j in range(cantNeuPorcapas[i]):
                Neu = neurona(self.cantInput)
                arrayNeuronas.append(Neu)
            nueva_capa = capa(arrayNeuronas)
            self.capas.append(nueva_capa)

    def train(self, maxEpoca, porcentaje):
        it = 0;
        porcetaje_aciertos = 0;
        while(it < maxEpoca & porcentaje_aciertos < porcentaje):
            # Loop de entrenamiento
            for i in range(len(self.capas)):
                fila = self.datos.iloc[i];
                entrada = np.array([fila.iloc[0], fila.iloc[1], fila.iloc[3]]);
                self.capas[0].forward_pass(entrada)
                
                for j in range(1,self.capas):
                    self.capas[j].forward_pass(self.capas[j-1].output)


            fila = self.datos.iloc[i]
            deseada = np.array(fila.iloc[2])
            self.capas[i].backward_pass(deseada)

            #calcular calibracion de pesos
            #primer delta
            fila = self.datos.iloc[i]
            deseada = np.array(fila.iloc[2])
            delta = self.capas[i].backward_pass(deseada)
            # Retropropagacion
            for i in range(len(self.capas)-1,0,-1):
                self.capas[i].backward_pass(delta, )

            #chequear sacar bias

            # y = self.capas[len(self.capas)-1].output;
            #chekin        
            
            it += 1

        



#definir variable de cantidad de salidas deseadas


pesobias = 0
it = 0;
epocaMax = 5;
datos = pd.read_csv("XOR_trn.csv");
datos["bias"] = -1;
porcentaje_aciertos = 0;
cantidad_entradas = len(datos);
eta = 0.005;

#config
#cin/cout
cantNeuPorcapas = [3,2,1]; #neuronas por capa
pesos = np.array();






                




while (it < epocaMax and porcentaje_aciertos < 90):
    # sum = 0;
    for i in range(cantidad_entradas):
        fila = datos.iloc[i];
        entrada = np.array([fila.iloc[0], fila.iloc[1], fila.iloc[3]]);

        z = np.dot(pesos, entrada); 
        y = 1 if z>=0 else -1;

        error = fila.iloc[2] - y; 
        pesos = pesos + (eta/2)*error*entrada;



#     for i in range(cantidad_entradas):
#         fila = datos.iloc[i];
#         entrada = np.array([fila.iloc[0], fila.iloc[1], fila.iloc[3]]);

#         z = np.dot(pesos, entrada); 
#         y = 1 if z>=0 else -1;

#         if(y == fila.iloc[2]):
#             sum += 1;

    
#     porcentaje_aciertos = (sum/cantidad_entradas)*100; 
#     print("Época:", it, "Aciertos:", sum, "Porcentaje:", porcentaje_aciertos);
#     it += 1;
    
# print("Pesos finales:", pesos);
# pesos_df = pd.DataFrame([pesos]);
# pesos_df.to_csv("pesos.csv",index=False);

