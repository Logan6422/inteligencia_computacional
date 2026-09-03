import numpy as np
import capa as c
import pandas as pd
import matplotlib.pyplot as plt

class red:
    def __init__(self, eta, cantEntradas, arquitectura):
        self.capas = [];
        self.eta = eta;

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


    def graficar(self,datos,cantidad_rectas):
        if not hasattr(self,"fig"):
            plt.ion();
            self.fig,self.ax=plt.subplots();
            self.fig.canvas.mpl_connect("close_event",self.cerrar_grafico);
        self.ax.clear();

        x1 = np.linspace(-2,2,100);
        primera_capa = self.capas[0];

        for i in range(cantidad_rectas):
            neurona = primera_capa.lista_neuronas[i];
            w1 = neurona.pesos[0];
            w2 = neurona.pesos[1];
            wb = neurona.pesosBias;

            if abs(w2) > 1e-8:
                x2 = (wb-w1*x1)/w2;
                self.ax.plot(x1,x2);
            else:
                if abs(w1) > 1e-8:
                    x_vertical = wb/w1;
                    self.ax.axvline(x_vertical);

        datos_pos = datos[datos.iloc[:,2] == 1];
        datos_neg = datos[datos.iloc[:,2] == -1];

        self.ax.scatter(datos_pos.iloc[:,0],datos_pos.iloc[:,1]);
        self.ax.scatter(datos_neg.iloc[:,0],datos_neg.iloc[:,1]);

        self.ax.set_title("Entrenamiento");
        self.ax.set_xlabel("x1");
        self.ax.set_ylabel("x2");

        #concent
        self.ax.set_xlim(-0.5,1.5);
        self.ax.set_ylim(-0.5,1.5);


        #xor
        # self.ax.set_xlim(-2,2);
        # self.ax.set_ylim(-2,2);

        self.fig.canvas.draw_idle();
        self.fig.canvas.flush_events();

        plt.pause(0.001);

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

                #la neurona agrega internamente el bias
                entrada = np.array([fila.iloc[0], fila.iloc[1]]);
                deseada = fila.iloc[2];

                self.forward_pass(entrada);
                self.backward_pass(deseada);
                self.actualizar_pesos_red();
           
            aciertos = 0;
            for i in range(len(datosEntrenamiento)):
                fila = datosEntrenamiento.iloc[i];

                entrada = np.array([fila.iloc[0], fila.iloc[1]]);
                deseada = fila.iloc[2];

                salida = self.forward_pass(entrada);

                prediccion = 1 if salida >= 0 else -1;
                if prediccion == deseada:
                    aciertos += 1;

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

            

def graficar_curvas(resultados_por_eta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,4));

    for eta, hist in resultados_por_eta.items():
        ax1.plot(hist["epocas"], hist["error_cuadratico"], label="eta=" + str(eta));
        ax2.plot(hist["epocas"], hist["error_clasificacion"], label="eta=" + str(eta));

    ax1.set_title("Error cuadrático total (ξ)");
    ax1.set_xlabel("épocas");
    ax1.set_ylabel("ξ");
    ax1.legend();
    ax1.grid(True);

    ax2.set_title("Error de clasificación");
    ax2.set_xlabel("épocas");
    ax2.set_ylabel("%");
    ax2.legend();
    ax2.grid(True);

    plt.tight_layout();
    plt.show();