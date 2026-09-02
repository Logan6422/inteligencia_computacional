import numpy as np
import capa as c
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        entradaIteracion = input
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

    # def cerrar_grafico(self,event):
    #     self.grafico_cerrado=True;

    def graficar_final(self, datos):

        plt.figure(figsize=(8, 6));

        for i in range(len(datos)):
            fila = datos.iloc[i];
            entrada = np.array(fila.iloc[:self.cantEntradas], dtype=float);
            deseada = np.array(fila.iloc[self.cantEntradas:], dtype=float);
            salida = np.array(self.forward_pass(entrada));

            if np.array_equal(deseada, [-1, -1, 1]):
                clase_real = "Setosa";
            elif np.array_equal(deseada, [-1, 1, -1]):
                clase_real = "Versicolor";
            elif np.array_equal(deseada, [1, -1, -1]):
                clase_real = "Virginica";

            prediccion = np.argmax(salida);

            if prediccion == 2:
                clase_predicha = "Setosa";
            elif prediccion == 1:
                clase_predicha = "Versicolor";
            else:
                clase_predicha = "Virginica";

            acierto = clase_real == clase_predicha;
            x = fila.iloc[0];
            y = fila.iloc[1];

            if clase_real == "Setosa":
                color = "red";
            elif clase_real == "Versicolor":
                color = "green";
            else:
                color = "blue";

            if acierto:
                marcador = "o";
            else:
                marcador = "x";

            plt.scatter(x, y, c=color, marker=marcador, s=70);

        plt.xlabel("x1");
        plt.ylabel("x2");
        plt.title("Distribución y clasificación final");
        plt.xlim(datos.iloc[:, 0].min() - 0.2, datos.iloc[:, 0].max() + 0.2);
        plt.ylim(datos.iloc[:, 1].min() - 0.2, datos.iloc[:, 1].max() + 0.2);

        plt.grid(True);

        leyenda = [
            Line2D([0], [0], marker="o", color="w",
                markerfacecolor="red", markersize=8,
                label="Setosa"),

            Line2D([0], [0], marker="o", color="w",
                markerfacecolor="green", markersize=8,
                label="Versicolor"),

            Line2D([0], [0], marker="o", color="w",
                markerfacecolor="blue", markersize=8,
                label="Virginica")
        ];
        plt.legend(handles=leyenda);

    def graficar_zona(self, datos, titulo="Zona de decisión"):
        primera = not hasattr(self, "fig_zona");
        if primera:
            self.fig_zona, self.ax_zona = plt.subplots(figsize=(8, 6));

        self.ax_zona.clear();
        margen = 0.2;
        x_min = datos.iloc[:, 0].min() - margen;
        x_max = datos.iloc[:, 0].max() + margen;
        y_min = datos.iloc[:, 1].min() - margen;
        y_max = datos.iloc[:, 1].max() + margen;

        x = np.linspace(x_min, x_max, 100);
        y = np.linspace(y_min, y_max, 100);
        X, Y = np.meshgrid(x, y);
        Z = np.zeros_like(X);

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                entrada = np.array([X[i, j], Y[i, j]]);
                salida = self.forward_pass(entrada);
                Z[i, j] = 1 if salida[0] >= 0 else -1;

        self.ax_zona.contourf(X, Y, Z, levels=[-1, 0, 1], colors=["lightcoral", "lightblue"], alpha=0.4);

        datos_pos = datos[datos.iloc[:, 2] == 1];
        datos_neg = datos[datos.iloc[:, 2] == -1];

        self.ax_zona.scatter(datos_pos.iloc[:, 0], datos_pos.iloc[:, 1], c='green', marker='o', edgecolor='black', s=60, label='Clase 1');
        self.ax_zona.scatter(datos_neg.iloc[:, 0], datos_neg.iloc[:, 1], c='purple', marker='x', s=80, linewidth=2, label='Clase -1');

        self.ax_zona.set_title(titulo);
        self.ax_zona.set_xlabel("x1");
        self.ax_zona.set_ylabel("x2");
        self.ax_zona.legend();
        self.ax_zona.grid(True);

        self.fig_zona.canvas.draw();
        self.fig_zona.canvas.flush_events();
        plt.pause(0.5 if primera else 0.001);   # la 1ra vez necesita más tiempo para crear y pintar la ventana

    def entrenar(self, printIt, printFinal, graficar, datosEntrenamiento, maxEpocas, porcentajeObjetivo):
            it = 0;
            porcentaje = 0;

            hist_error_cuadratico = [];
            hist_error_clasificacion = [];

            while it < maxEpocas and porcentaje < porcentajeObjetivo:
                error_cuadratico_epoca = 0;
                # Loop de entrenamiento

                for i in range(len(datosEntrenamiento)):
                    fila = datosEntrenamiento.iloc[i];
                    entrada = np.array(fila.iloc[:self.cantEntradas], dtype=float);
                    deseada = np.array(fila.iloc[self.cantEntradas:],dtype=float);
                    salida = self.forward_pass(entrada);
                    e = deseada - np.array(salida);
                    error_cuadratico_epoca += 0.5 * np.sum(e**2);
                    self.backward_pass(deseada);
                    self.actualizar_pesos_red();

                # Loop de evaluación (aciertos)

                aciertos = 0;

                for i in range(len(datosEntrenamiento)):
                    fila = datosEntrenamiento.iloc[i];
                    entrada = np.array(fila.iloc[:self.cantEntradas], dtype=float);
                    deseada = np.array(fila.iloc[self.cantEntradas:],dtype=float);
                    salida = self.forward_pass(entrada);
                    prediccion = np.where(np.array(salida) >= 0, 1, -1);

                    if np.array_equal(prediccion, deseada):
                        aciertos += 1;

                porcentaje = (aciertos / len(datosEntrenamiento)) * 100;
                error_clasificacion_epoca = 100 - porcentaje;

                hist_error_cuadratico.append(error_cuadratico_epoca);
                hist_error_clasificacion.append(error_clasificacion_epoca);

                if(printIt):
                    print("Epoca:", it, "Aciertos:", aciertos, "Porcentaje:", round(porcentaje,2), "Error cuad:", round(error_cuadratico_epoca, 4));

                if(graficar and self.cantEntradas == 2 and it % 10 == 0): # graficar cada N epocas
                    self.graficar_zona(datosEntrenamiento, "Entrenamiento - época " + str(it));

                it += 1;

            if(printFinal):
                self.graficar_final(datosEntrenamiento);

            plt.ioff();
            plt.show();

            return {
                "epocas": list(range(it)),
                "error_cuadratico": hist_error_cuadratico,
                "error_clasificacion": hist_error_clasificacion
            }

def graficar_curvas(resultados_por_eta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6));

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