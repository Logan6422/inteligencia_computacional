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

            self.fig, self.ax = plt.subplots(figsize=(8, 6))

            self.fig.canvas.mpl_connect(
                "close_event",
                self.cerrar_grafico
            )

        self.ax.clear()

        # -------------------------------------------------
        # x3 y x4 fijos
        # -------------------------------------------------

        if x3 is None:
            x3 = datos.iloc[:, 2].mean()

        if x4 is None:
            x4 = datos.iloc[:, 3].mean()

        # -------------------------------------------------
        # Evaluamos todos los patrones
        # -------------------------------------------------

        for i in range(len(datos)):

            fila = datos.iloc[i]

            entrada = np.array(
                fila.iloc[:self.cantEntradas],
                dtype=float
            )

            deseada = np.array(
                fila.iloc[self.cantEntradas:],
                dtype=float
            )

            salida = np.array(
                self.forward_pass(entrada)
            )

            # ---------------------------------------------
            # Determinamos clase real
            # ---------------------------------------------

            if np.array_equal(deseada, [-1, -1, 1]):

                clase_real = "Setosa"

            elif np.array_equal(deseada, [-1, 1, -1]):

                clase_real = "Versicolor"

            elif np.array_equal(deseada, [1, -1, -1]):

                clase_real = "Virginica"

            # ---------------------------------------------
            # Determinamos clase predicha
            # ---------------------------------------------

            prediccion = np.argmax(salida)

            if prediccion == 2:

                clase_predicha = "Setosa"

            elif prediccion == 1:

                clase_predicha = "Versicolor"

            else:

                clase_predicha = "Virginica"

            # ---------------------------------------------
            # ¿Acertó?
            # ---------------------------------------------

            acierto = clase_real == clase_predicha

            # ---------------------------------------------
            # Posición
            # ---------------------------------------------

            x = fila.iloc[0]
            y = fila.iloc[1]

            # ---------------------------------------------
            # Color según clase REAL
            # ---------------------------------------------

            if clase_real == "Setosa":

                color = "red"

            elif clase_real == "Versicolor":

                color = "green"

            else:

                color = "blue"

            # ---------------------------------------------
            # Marcador según acierto/error
            # ---------------------------------------------

            if acierto:

                marcador = "o"

            else:

                marcador = "x"

            self.ax.scatter(
                x,
                y,
                c=color,
                marker=marcador,
                s=70
            )

        # -------------------------------------------------
        # Configuración
        # -------------------------------------------------

        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")

        self.ax.set_title(
            "Distribución y clasificación"
        )

        self.ax.set_xlim(
            datos.iloc[:, 0].min() - 0.2,
            datos.iloc[:, 0].max() + 0.2
        )

        self.ax.set_ylim(
            datos.iloc[:, 1].min() - 0.2,
            datos.iloc[:, 1].max() + 0.2
        )

        self.ax.grid(True)

        self.fig.tight_layout()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    def entrenar(self, printIt, printFinal, graficar, datosEntrenamiento, maxEpocas, porcentajeObjetivo):
            it = 0;
            porcentaje = 0;
            
            # NUEVO: Listas para guardar el historial de errores por época
            hist_error_cuadratico = []
            hist_error_clasificacion = []

            while it < maxEpocas and porcentaje < porcentajeObjetivo:
                error_cuadratico_epoca = 0
                
                # Loop de entrenamiento
                for i in range(len(datosEntrenamiento)):
                    fila = datosEntrenamiento.iloc[i];

                    entrada = np.array(fila.iloc[:self.cantEntradas], dtype=float);
                    deseada = np.array(fila.iloc[self.cantEntradas:],dtype=float);

                    salida = self.forward_pass(entrada);
                    
                    # NUEVO: Calcular error cuadrático instantáneo: xi = 1/2 * sum(e_j^2)
                    e = deseada - np.array(salida)
                    error_cuadratico_epoca += 0.5 * np.sum(e**2)
                    
                    self.backward_pass(deseada);
                    self.actualizar_pesos_red();
            
                # Loop de evaluación (aciertos)
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
                error_clasificacion_epoca = 100 - porcentaje # NUEVO
                
                # NUEVO: Guardar en los historiales
                hist_error_cuadratico.append(error_cuadratico_epoca)
                hist_error_clasificacion.append(error_clasificacion_epoca)

                if(printIt):
                    print("Epoca:", it, "Aciertos:", aciertos, "Porcentaje:", round(porcentaje,2), "Error cuad:", round(error_cuadratico_epoca, 4));

                if graficar:
                    self.grafico_cerrado = False
                    self.graficar(datosEntrenamiento)
                   
                it += 1;

            if(printFinal):
                self.graficar(datosEntrenamiento)

            plt.ioff()
            plt.show()
            # NUEVO: Devolver el diccionario con los historiales para graficar
            return {
                "epocas": list(range(it)),
                "error_cuadratico": hist_error_cuadratico,
                "error_clasificacion": hist_error_clasificacion
            }


    # NUEVO: Función independiente al final de red.py (fuera de la clase)
    def graficar_curvas(resultados_por_eta):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for eta, hist in resultados_por_eta.items():
            ax1.plot(hist["epocas"], hist["error_cuadratico"], label=f"eta={eta}")
            ax2.plot(hist["epocas"], hist["error_clasificacion"], label=f"eta={eta}")

        ax1.set_title("Error cuadrático total (ξ)")
        ax1.set_xlabel("Épocas")
        ax1.set_ylabel("ξ")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Error de clasificación")
        ax2.set_xlabel("Épocas")
        ax2.set_ylabel("%")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        
                
    
    