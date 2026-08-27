import numpy as np
import capa as c
import pandas as pd
import matplotlib.pyplot as plt

class red:
    def __init__(self, eta, cantEntradas, arquitectura):
        self.capas = [];
        self.eta = eta;
        self.cantEntradas = cantEntradas;

        cantEntradaIteracion = cantEntradas;
        for i in range(len(arquitectura)):
            new_capa = c.capa(arquitectura[i],cantEntradaIteracion);
            self.capas.append(new_capa);
            cantEntradaIteracion = len(new_capa.lista_neuronas);
            


    def forward_pass(self, input):
        entradaIteracion = input;
        for i in range(len(self.capas)):
            entradaIteracion = self.capas[i].forward_pass(entradaIteracion);

        # print(self.capas[-1].output);
        return self.capas[-1].output; #salida final (lo dejo en vector por si puede haber mas de una salida)

    def backward_pass(self, deseada):
        #delta de la capa de salida
        self.capas[-1].backward_final(deseada);
        
        #desde la penultima capa hasta la primera
        for i in range(len(self.capas) - 2, -1, -1):
            capa_sig = self.capas[i + 1];
            delta_sig = capa_sig.lista_deltas;
            pesos_sig = [];

            for j in range(len(self.capas[i].lista_neuronas)):
                pesos_neurona = [];
                
                for k in range(len(capa_sig.lista_neuronas)):
                    peso = capa_sig.lista_neuronas[k].pesos[j];
                    pesos_neurona.append(peso);

                pesos_sig.append(pesos_neurona);

            self.capas[i].backward_oculta(pesos_sig, delta_sig);

    def actualizar_pesos_red(self):
        
        for i in range(len(self.capas)):
            self.capas[i].actualizar_pesos_capa(self.eta);
    

    def cerrar_grafico(self,event):
        self.grafico_cerrado=True;


    def graficar(self, datos, x3=None, x4=None):

        if not hasattr(self, "fig"):

            plt.ion()

            self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))

            self.fig.canvas.mpl_connect(
                "close_event",
                self.cerrar_grafico
            )

        # Si no especificamos x3 y x4,
        # usamos el promedio de los datos
        if x3 is None:
            x3 = datos.iloc[:, 2].mean()

        if x4 is None:
            x4 = datos.iloc[:, 3].mean()

        # Rango de x1 y x2 según los datos
        x1 = np.linspace(
            datos.iloc[:, 0].min(),
            datos.iloc[:, 0].max(),
            100
        )

        x2 = np.linspace(
            datos.iloc[:, 1].min(),
            datos.iloc[:, 1].max(),
            100
        )

        X, Y = np.meshgrid(x1, x2)

        # Una matriz para cada salida
        Z = [
            np.zeros_like(X),
            np.zeros_like(X),
            np.zeros_like(X)
        ]

        # Calculamos la salida de la red para cada punto
        for i in range(X.shape[0]):

            for j in range(X.shape[1]):

                entrada = np.array([
                    X[i, j],
                    Y[i, j],
                    x3,
                    x4
                ])

                salida = self.forward_pass(entrada)

                Z[0][i, j] = salida[0]
                Z[1][i, j] = salida[1]
                Z[2][i, j] = salida[2]

        # Clases reales
        deseadas = datos.iloc[:, 4:7].to_numpy()

        # Cada salida tiene su propio gráfico
        for k in range(3):

            self.ax[k].clear()

            self.ax[k].contourf(
                X,
                Y,
                Z[k],
                levels=[-1, 0, 1],
                alpha=0.3
            )

            # Dibujamos los puntos según la clase real
            for i in range(len(datos)):

                x = datos.iloc[i, 0]
                y = datos.iloc[i, 1]

                if deseadas[i, k] == 1:
                    self.ax[k].scatter(
                        x,
                        y,
                        marker="o"
                    )
                else:
                    self.ax[k].scatter(
                        x,
                        y,
                        marker="x"
                    )

            self.ax[k].set_xlabel("x1")
            self.ax[k].set_ylabel("x2")

            self.ax[k].set_title(
                "Salida y" + str(k + 1)
            )

            self.ax[k].set_xlim(
                datos.iloc[:, 0].min()-2,
                datos.iloc[:, 0].max()+2
            )

            self.ax[k].set_ylim(
                datos.iloc[:, 1].min()-2,
                datos.iloc[:, 1].max()+2
            )

        self.fig.tight_layout()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.0001)

    def visualizador_final(self, it, aciertos, porcentaje, datos):
        print("Pesos finales:");
        for i in range(len(self.capas)):
            print("Capa", i);
            for j in range(len(self.capas[i].lista_neuronas)):
                neurona = self.capas[i].lista_neuronas[j];
                print("Neurona", j, ":", neurona.pesos,"Bias:", neurona.pesosBias);
            print("\n");
        print("\n");

        print("/////////////////////////////////////////////////////");
        print("\n");

        print("Aciertos:", aciertos);
        print("\n");

        print("Porcentaje:", porcentaje);
        print("\n");

        print("Epocas:", it);
        print("\n");

        #Visualizacion final
        plt.ion();
        self.fig, self.ax = plt.subplots();
        cantidad_rectas = len(self.capas[0].lista_neuronas);

        self.graficar(datos, cantidad_rectas);
        self.graficar_zona(datos);

        self.ax.set_title("Resultado final");
        self.fig.canvas.draw_idle();
        self.fig.canvas.flush_events();

        print("Grafico final");
        plt.ioff();
        plt.show();

    def graficar_zona(self,datos):
        x = np.linspace(-2,2,100);
        y = np.linspace(-2,2,100);

        X,Y = np.meshgrid(x,y);
        Z = np.zeros_like(X);

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                entrada = np.array([X[i,j],Y[i,j]]);
                salida = self.forward_pass(entrada); 

                if salida>=0:
                    Z[i,j] = 1;

                else:
                    Z[i,j] = -1;

        self.ax.contourf(X,Y,Z, levels=[-1,0,1], colors=["red", "blue"], alpha=0.3);


        self.ax.set_title("Zona de decisión");

        self.fig.canvas.draw_idle();
        self.fig.canvas.flush_events();

        plt.pause(0.0001);

    def entrenar(self, printIt, printFinal, graficar, datosEntrenamiento, maxEpocas, porcentajeObjetivo):
        it = 0;
        porcentaje = 0;

        while it < maxEpocas and porcentaje < porcentajeObjetivo:
            for i in range(len(datosEntrenamiento)):
                fila = datosEntrenamiento.iloc[i];

                # print(entrada);
                #la neurona agrega internamente el bias
                # entrada = np.array([fila.iloc[0], fila.iloc[1]]);

                entrada = np.array(fila.iloc[:self.cantEntradas], dtype=float);
                deseada = np.array(fila.iloc[self.cantEntradas:],dtype=float);
                # deseada = fila.iloc[2];

                self.forward_pass(entrada);
                self.backward_pass(deseada);
                self.actualizar_pesos_red();
           
            aciertos = 0;
            for i in range(len(datosEntrenamiento)):
                fila = datosEntrenamiento.iloc[i];

                entrada = np.array(fila.iloc[:self.cantEntradas], dtype=float);
                deseada = np.array(fila.iloc[self.cantEntradas:],dtype=float);

                salida = self.forward_pass(entrada);
                prediccion = np.where(np.array(salida) >= 0, 1, -1)
                if np.array_equal(prediccion, deseada):
                    aciertos += 1

            porcentaje = (aciertos / len(datosEntrenamiento)) * 100;

            if(printIt):
                print("Epoca:", it,"Aciertos:", aciertos,"Porcentaje:", porcentaje);

            if(graficar):
                self.grafico_cerrado=False;
                for j in range(len(self.capas[0].lista_neuronas)):
                    self.graficar(datosEntrenamiento,j+1);

        
            it += 1;

        if(printFinal):
            self.visualizador_final(it,aciertos,porcentaje,datosEntrenamiento);

            

