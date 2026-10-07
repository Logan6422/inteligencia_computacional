Aprendizaje no supervisado "clustering" (no neuronal)
- Particionales 
    - k-medias 
    - k-medoides(PAM) solucion cuando no podemos obtener centroides, por que no podemos obtener centroides? ej texto, tenes frases, medis distancias entre palabras o frases por ej, pero no podes promediar frases (centroides)
- Jerarquicos
    - Aglomerativos(bottom up) agrupa los que estan mas cerca y arma claster, y a su vez agrupa por claster segun cercania, va construyendo clusters mas grandes.
    - Divisivos (top-down)
- Basados en densidad (hacen crecer la zona hasta cierto limite)
    - DBSCAN
    - OPTICS
    - Mean Shift
- Basados en modelos estadisticos
    - Mezclas Gaussianas (GMM)
    - Maximacion de la esperanza (EM para clustering)
- Y mas: basados en grafos, custering espectral, mapas auto organizativos, basados en aprendizaje profundo...

-> clustering jerarquico agarro los dos que estan mas cerca y hago un claster, agarro los que estan cerca y los voy agregando y agregando el cluster (video en las diapositivas)
-> DBSCAN: se basa en elegirp rimero un dato cualquiera, y apartir de ese(al azar) empiezo a buscar los que estan mas cercanos y siempre que encuentor uno que esta por ej distancia minima, entonces encuentro uno que esta a esa distancia y lo agrego al cluster, y apartir de esos dos voy agregando en base a la distancia y hago crecer el cluster, hasta cuando no encuentre nada de esa distancia minima, y ahora tomo un punto al azar que no sea ninguno de los que ya tengo y empiezo a hacer lo mismo, cuando ya hice crecer todo lo que podia, se van a ver puntos que quedan fuera y que no quedaron en ningun cluster.

* Clasificacion de las metricas de clustering
    - Metricas Internas: eval la calidad usando solo la info de los datos y la asignacion a grupos.
        - Compactitud, inercia o suma de los cuadrados intra cluster
        - Separacion
        - Davis-Bouldin
        - Dunn
        - Silhouette
    - Metricas externas: comparan los agrupamientos obtenidos con una referencia o etiqueta correcta.
        - Fowlkes- Mallows, Jaccard, Dice-sorensen, F1
        - Indice Rand(RI) e indice Rand ajustado (ARI)
        - Informacion mutua (MI) e MI normalizada (NMI)
        - Calinski-Harabasz
        - Homogeneidad, completitud y medida V

- Metricas Internas
    - Compactitud, que tan compactos son los clusters formados, que tan desparramados estan(promedio entre centroide y los patrones)
    - Separacion, que tan separados estan (promedio de distancia entre centroides)
    - Separacion media
    - Importa la Relacion entre compactitud y separacion, que sea compacto y esten bien separados
    - Indice de davies-bouldin, si son muy compactos numerador chiquito, si estan separados numerador grande, barre sobre todos los cluster, lo comparo con todos los demas de todos esos me quedo el maximo (el peor) ahora agarro el siguiente y hago lo mismo, al final hago un promedio de todo eso, promedio de las peores situaciones.
    - Indice de dunn, no usa los centroides, recorre todo los patrones x que hay (omega es el cluster al que pertenece), se busca el par de puntos del cluster 1 y 2 que tienen la menor distancia(el minimo) eso lo hago para todos los grupos y de todos los grupos me quedo con el minimo(el peor caso), en el denominador hago lo inverso, busco la distancia maxima que da una idea de que tan desparramados estan en esa distancia
    - Silhouette para cada punto x_i del cluster omega_j

- Metricas Externas, comparacion con etiquetas reales, de referencia, clases conocidas. Comparacion entre diferentes soluciones de clustering.
    - Matriz de contigencia: contigencia entre los elementos de los clusters. Donde esta la mayor coincidencia entre clusters 
    - Comparacion por pares 
        - Indice de Rand
        - Indice Rand ajustado(ARI)
        - Indice de jaccard
        - Indice de Dice-Sorensen
        - F1
        - Fowlkes-Mallows
    - Teoria de la informacion
        - Informacion mutua(MI)

- como determinar el k optimo?
    - Metodo del codo(elbow) tomar alguna metrica (ej compartitud pero no necesariamente eso) ir aumentando k y viendo cuando la curva se planca, ya no tiene sentido aumentar el k porque ya llego al limite, te quedas con ese k, si agregas mas k estas perdiendo generalizacion, llega un punto que si k es muy grande empieza a perder generalizacion

practica 4 - usar indices no listas, no asignar fisicamente