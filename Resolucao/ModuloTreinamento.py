import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import preprocessing
import math
from pickle import dump


dados = pd.read_csv('HousingData.csv', sep=',')

medianas = dados.median()
dados = dados.fillna(medianas)

normalizador = preprocessing.MinMaxScaler()

dados_normalizador = normalizador.fit_transform(dados)

dump(normalizador, open('modelo_normalizador.pkl', 'wb'))

distorcoes = []
K = range(1, 100)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_normalizador)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizador, cluster_model.cluster_centers_, 'euclidean'), axis=1
            )/ dados_normalizador.shape[0]
        )    
    )

x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0)**2 + (xn - x0)**2)
    distancias.append(numerador / denominador)

n_cluster_otimo = K[distancias.index(max(distancias))]
print("Número ótimo de clusters:", n_cluster_otimo)

modelo_cluster = KMeans(n_clusters=n_cluster_otimo, random_state=42, n_init=10).fit(dados_normalizador)

dump(modelo_cluster, open('modeloHousing_cluster.pkl', 'wb'))
