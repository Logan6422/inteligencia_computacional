import numpy as np
import random

class Kmedias:
    def __init__(self, n_cluster, random_states, datos, max_epocas): 
        self.k = n_cluster;
        self.random_states = random_states;
        self.datos = datos;
        self.max_epocas = max_epocas;
        self.nro_patrones = datos.shape[0];
        self.centroides = np.empty(shape=(self.k, datos.shape[1]));
        self.asignacion = [];
        random.seed(self.random_states)  # AGREGADO: inicializar la semilla
        self.inicialize();

    def inicialize(self):
        # inicializar pesos
        indices = np.arange(self.nro_patrones);
        random.shuffle(indices);
        indice = 0;
        for i in range(self.k):
            self.centroides[i] = self.datos[indices[i]];


    def fit(self):
        # etapa 1 asignacion de datos 
        epoca = 0;
        while(epoca < self.max_epocas):
            print(f"Epoca: {epoca}")
            self.asignacion = [] 
            for entrada in range(self.nro_patrones):
                # Seleccion de centroide ganador
                resta = np.subtract(self.datos[entrada], self.centroides);
                normas = np.linalg.norm(resta, ord=2, axis=1);
                self.asignacion.append(np.argmin(normas));

            for centroide in range(self.k):
                suma = 0;
                cant = 0;
                for i in range(self.nro_patrones):  #recorre los patrones
                    if self.asignacion[i] == centroide:
                        suma += self.datos[i];
                        cant += 1;

                if(cant != 0): 
                    self.centroides[centroide] = suma / cant; #actualiza centroides
            
            epoca += 1;

    def cluster_centers(self):
        return self.centroides;

    def labels(self):
        return self.asignacion;