import numpy as np

#b = 1
def sigmoidea(z):
    return (2/(1 + np.exp(-z)))-1

def derivada_sigmoidea(y):
    return 0.5*(1 + y)*(1-y)

class neurona:
    def __init__(self, cantEntradas):
        self.pesos = np.random.uniform(-0.5,0.5,cantEntradas);
        self.pesosBias = np.random.uniform(-0.5, 0.5, 1)[0];
        self.delta = 0;
        self.y = 0;
        self.z = 0;

    def forward(self,input):
        entrada = input.copy(); #agrego el -1 del bias
        entrada.append(-1);
        pesos_completo = np.append(self.pesos,self.pesosBias);
        self.z = np.dot(pesos_completo, entrada);
        self.y = sigmoidea(self.z);
        return self.y;

    def backward(self, deseada):
        error = (deseada - self.y);
        derivada = derivada_sigmoidea(self.y);
        self.delta = error*derivada;

    def backward_oculta(self, deltas_sig, pesos_sig):
        derivada = derivada_sigmoidea(self.y);

        suma = 0;
        for i in range(len(deltas_sig)):
            suma += deltas_sig[i] * pesos_sig[i];

        self.delta = suma * derivada;
        
