import numpy as np
import neurona as n

def derivada_sigmoidea(y):
    return 0.5*(1 + y)*(1-y)

class capa:
    def __init__(self, cant_neuronas, cant_entradas):
        self.lista_neuronas = [];
        self.lista_deltas = [];
        for i in range(cant_neuronas):
            neu = n.neurona(cant_entradas);
            self.lista_neuronas.append(neu);



    def forward_pass(self, input):
        self.output = [];
        for i in range(len(self.lista_neuronas)):
            res = self.lista_neuronas[i].forward(input);
            self.output.append(res);

        return self.output;

    def backward_final(self, deseada):
        self.lista_deltas = [];
        for i in range(len(self.lista_neuronas)):
            self.lista_neuronas[i].backward(deseada[i]);
            delta = self.lista_neuronas[i].delta;
            self.lista_deltas.append(delta);


    def backward_oculta(self, pesos_sig, delta_sig):
        self.lista_deltas = [];
        for i in range(len(self.lista_neuronas)):
            self.lista_neuronas[i].backward_oculta(delta_sig, pesos_sig[i]);
            delta = self.lista_neuronas[i].delta;
            self.lista_deltas.append(delta);


    def actualizar_pesos_capa(self, eta):
        for i in range(len(self.lista_neuronas)):
            self.lista_neuronas[i].actualizar_pesos(eta);