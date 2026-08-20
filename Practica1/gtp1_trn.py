import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



it = 0;
epocaMax = 5;
datos = pd.read_csv("XOR_trn.csv");
datos["bias"] = -1;
porcentaje_aciertos = 0;
pesos = np.random.uniform(-0.5,0.5,3);
cantidad_entradas = len(datos);
eta = 0.005;
print("Pesos iniciales:", pesos)

plt.ion(); #modo interactivo
fig, ax = plt.subplots();

ax.set_title("Entrenamiento")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

ax.axhline(0)
ax.axvline(0)
datos_pos = datos[datos.iloc[:, 2] == 1];
datos_neg = datos[datos.iloc[:, 2] == -1];
ax.scatter(datos_pos.iloc[:, 0], datos_pos.iloc[:, 1]);
ax.scatter(datos_neg.iloc[:, 0], datos_neg.iloc[:, 1]);
x1 = np.linspace(-2, 2, 100);
x2 = (pesos[2]/pesos[1]) - (pesos[0]/pesos[1])*x1;
linea, = ax.plot(x1, x2, color="red");


while (it < epocaMax and porcentaje_aciertos < 90):
    sum = 0;
    for i in range(cantidad_entradas):
        fila = datos.iloc[i];
        entrada = np.array([fila.iloc[0], fila.iloc[1], fila.iloc[3]]);

        z = np.dot(pesos, entrada); 
        y = 1 if z>=0 else -1;

        error = fila.iloc[2] - y; 
        pesos = pesos + (eta/2)*error*entrada;

        if i % 10 == 0:
            x2 = (pesos[2]/pesos[1]) - (pesos[0]/pesos[1])*x1;
            linea.set_ydata(x2);

            fig.canvas.draw_idle();
            fig.canvas.flush_events();
            plt.pause(0.001);
    
    #grafico una vez por epoca
    # x1 = np.linspace(-2, 2, 100);
    # x2 = (pesos[2]/pesos[1]) - (pesos[0]/pesos[1])*x1;
    # linea.set_ydata(x2)
    # fig.canvas.draw();
    # fig.canvas.flush_events();
    # plt.pause(0.1);
    # plt.pause(0.3);     


    for i in range(cantidad_entradas):
        fila = datos.iloc[i];
        entrada = np.array([fila.iloc[0], fila.iloc[1], fila.iloc[3]]);

        z = np.dot(pesos, entrada); 
        y = 1 if z>=0 else -1;

        if(y == fila.iloc[2]):
            sum += 1;

    
    porcentaje_aciertos = (sum/cantidad_entradas)*100; 
    print("Época:", it, "Aciertos:", sum, "Porcentaje:", porcentaje_aciertos);
    it += 1;
    
print("Pesos finales:", pesos);
pesos_df = pd.DataFrame([pesos]);
pesos_df.to_csv("pesos.csv",index=False);

plt.ioff();
plt.show(); 