import numpy as np
import capa as c

class red:
    def __init__(self, eta, cantEntradas, arquitectura):
        self.capas = [];
        self.eta = eta;
        # self.datos = 0 (pandas csv)
        deseada = [];

        cantEntradaIteracion = cantEntradas;
        for i in range(len(arquitectura)):
            new_capa = c.capa(arquitectura[i],cantEntradaIteracion);
            self.capas.append(new_capa);
            cantEntradaIteracion = len(new_capa.lista_neuronas);


    def forward_pass(self, input):
        entradaIteracion = input;
        for i in range(len(self.capas)):
            entradaIteracion = self.capas[i].forward_pass(entradaIteracion);

        return self.capas[-1].output[0]; #salida final (lo dejo en vector por si puede haber mas de una salida)

    def backward_pass(self, deseada):
        #delta salida
        self.capas[-1].backward_final(deseada)

        for i in range(len(self.capas[i].lista_neuronas) - 2, -1, -1):
            capa_sig = self.capas[i + 1]
            delta_sig = capa_sig.lista_deltas

            pesos_sig = []
            for j in range(len(self.capas[i])):
                pesos_neurona = []

                for k in range(len(capa_sig.lista_neuronas)):
                    peso = capa_sig.lista_neuronas[k].pesos[j]
                    pesos_neurona.append(peso)

                pesos_sig.append(pesos_neurona)

            # Calcular deltas de la capa actual
            self.capas[i].backward_oculta(pesos_sig, delta_sig)