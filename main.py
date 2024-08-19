import pandas as pd
import numpy
import numpy as np

#columns = ["DateTime","Temperature","Humidity","Wind Speed","general diffuse flows","diffuse flows","Zone 1 Power Consumption","Zone 2  Power Consumption","Zone 3  Power Consumption"]

df = pd.read_csv('Tetuan City power consumption.csv')

print(df.head())
df.info()

duplicate = df.duplicated()

filas_duplicadas = df[duplicate]
print(filas_duplicadas)

vacio = df.isnull().sum()
print(vacio[vacio > 0])

#Filas vacias
filas_vacias = df[df.isnull().any(axis=1)]
print(filas_vacias)

df = df.drop(columns=['Zone 2  Power Consumption', 'Zone 3  Power Consumption'])

print(df.head())

'''
Normalización de los datos
Con el proposito de que los datos más grandes no influyan en los más pequeños.
'''
valor_min = df['Zone 1 Power Consumption'].min()
valor_max = df['Zone 1 Power Consumption'].max()
'''
Formula de normalización
x - min / (max - min)

Consumo total de zona 1 normalizada
'''
df['Zone 1 normalized'] = (df['Zone 1 Power Consumption'] - valor_min) / (valor_max - valor_min)

print(df[['Zone 1 Power Consumption','Zone 1 normalized']].head())

#Nomalizar Temperatura
valor_min = df['Temperature'].min()
valor_max = df['Temperature'].max()

df['Temperature normalized'] = (df['Temperature'] - valor_min) / (valor_max - valor_min)
print(df[['Temperature','Temperature normalized']].head())

#Nomalizar Humedad
valor_min = df['Humidity'].min()
valor_max = df['Humidity'].max()

df['Humidity normalized'] = (df['Humidity'] - valor_min) / (valor_max - valor_min)
print(df[['Humidity','Humidity normalized']].head())

print(df.head())

df.to_csv('Normalización de datos.csv', index=False)

'''
Editar csv eliminando comolunas no normalizadas
'''

df = df.drop(columns=['Zone 1 Power Consumption', 'Temperature', 'Humidity'])

df = df.round(3)

columnas_ordenadas = [
    'DateTime', 
    'Temperature normalized', 
    'Humidity normalized', 
    'Wind Speed', 
    'general diffuse flows', 
    'diffuse flows', 
    'Zone 1 normalized'
]
df = df[columnas_ordenadas]
print(df.head())
df.to_csv('Normalización de datos.csv', index=False)

'''
Implementación de la regresión lineal, debido a que mis datos no son clasificados
'''

samples = df[['Temperature normalized', 'Humidity normalized', 'Wind Speed', 'general diffuse flows', 'diffuse flows']].values.tolist()
y = df['Zone 1 normalized'].values.tolist()

__erros__ = []

def h(params, sample):
    # Calcular la hipótesis (producto punto de los parámetros y las muestras)
    acum = 0
    for i in range(len(params)):
        acum += params[i] * sample[i]
    return acum

def show_error(params, samples, y):
    # Calcular el error cuadrático medio
    global __erros__
    error_acum = 0
    for i in range(len(samples)):
        hyp = h(params, samples[i])
        error = hyp - y[i]
        error_acum += error ** 2
    mean_error_param = error_acum / (2 * len(samples))
    __erros__.append(mean_error_param)
    return mean_error_param

def GD(params, samples, y, alfa):
    # Gradiente descendiente
    m = len(samples)
    temp = [0] * len(params)
    for j in range(len(params)):
        acum = 0
        for i in range(m):
            error = h(params, samples[i]) - y[i]
            acum += error * samples[i][j]
        temp[j] = params[j] - (alfa / m) * acum
    return temp

def scaling(samples):
    # Escalado de las características
    samples_transposed = list(zip(*samples))
    scaled_samples = []
    for i in range(len(samples_transposed)):  # Ignorar la primera columna si está presente
        feature = list(samples_transposed[i])
        avg = sum(feature) / len(feature)
        max_val = max(feature)
        scaled_feature = [(x - avg) / max_val for x in feature]
        scaled_samples.append(scaled_feature)
    scaled_samples = list(zip(*scaled_samples))
    # Convertir las tuplas de nuevo a listas
    scaled_samples = [list(sample) for sample in scaled_samples]
    return scaled_samples

# Inicializar los parámetros
params = [0] * (len(samples[0]) + 1)  # n+1 porque añadimos la columna de unos
alfa = 0.01
epochs = 0

# Agregar una columna de unos (para el término independiente)
samples = [[1] + sample for sample in samples]

samples = scaling(samples)

errors = []

for i in range(len(samples)):
    if isinstance(samples[i], list):
        samples[i] = [1] + samples[i]
    else:
        samples[i] = [1, samples[i]]

print("Original samples:")
print(samples)

print("Original samples:")
print(samples)
while True:
    oldparams = list(params)
    print("params")
    params = GD(params, samples, y, alfa)
    show_error(params, samples, y)
    print("params")
    epochs += 1
    if oldparams == params or epochs == 2:  # Limitar a 1000 epochs para evitar bucle infinito
        print("samples:")
        print(samples)
        print("final params:")
        print(params)
        break

import matplotlib.pyplot as plt
# Visualizar la gráfica de errores
plt.plot(__erros__)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Error during Gradient Descent')
plt.show()