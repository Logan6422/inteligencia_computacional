import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import csv
from sklearn.cluster import KMeans


datos = np.loadtxt("iris81_trn.csv", delimiter=",")
x_iris_trn = datos[:, [0,1,2,3]]
y_iris_trn = datos[:, [4, 5 ,6]]

datos = np.loadtxt("iris81_tst.csv", delimiter=",")
x_iris_tst = datos[:, [0,1,2,3]]
y_iris_tst = datos[:, [4, 5 ,6]]

kmedias = KMeans(n_clusters=15, 
                init="random",
                random_state=67)

indices = kmedias.fit_predict(x_iris_trn)
print(indices)
print(kmedias.score(x_iris_trn, y_iris_trn))
