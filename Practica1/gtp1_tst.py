import numpy as np
import pandas as pd

it = 0;
epocaMax = 100;
datos = pd.read_csv("OR_50tst.csv");
datos["bias"] = -1;
porcentaje_aciertos = 0;
pesos = pd.read_csv("pesos.csv").to_numpy().flatten()
cantidad_entradas = len(datos);
sum = 0;

for i in range(cantidad_entradas):
        fila = datos.iloc[i];
        entrada = np.array([fila.iloc[0], fila.iloc[1], fila.iloc[3]]);
        
        z = np.dot(pesos, entrada); 
        y = 1 if z>=0 else -1;

        if(y == fila.iloc[2]):
            sum += 1;

porcentaje_aciertos = (sum/cantidad_entradas)*100; 
print("Pesos:", pesos, "Porcentaje aciertos: ", porcentaje_aciertos);

historico = pd.read_csv("resultados_historicos_OR_50.csv");
nueva_fila = pd.DataFrame(
    [[pesos[0], pesos[1], pesos[2], porcentaje_aciertos]],
    columns=["w1", "w2", "w0", "porcentaje"]
)
historico = pd.concat([historico, nueva_fila],ignore_index=True);
historico.to_csv("resultados_historicos_OR_50.csv",index=False);