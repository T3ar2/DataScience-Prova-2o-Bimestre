from pickle import load

modelo_normalizador = load(open('modelo_normalizador.pkl', 'rb'))
novaInstancia = load(open('modeloHousing_cluster.pkl', 'rb'))

nova_residencia = [[1.31842,0,8.14,0,0.538,5.63,96.4,3.8512,4,307,21,379.21,20.47,14.2]]

nova_residencia_normalizada = modelo_normalizador.transform(nova_residencia)

numero_do_cluster = novaInstancia.predict(nova_residencia_normalizada)

centroide_normalizado = [novaInstancia.cluster_centers_[numero_do_cluster[0]]]

centroide_desnormalizado = modelo_normalizador.inverse_transform(centroide_normalizado)
print("As medidas reais do centroide deste grupo são:", centroide_desnormalizado[0])