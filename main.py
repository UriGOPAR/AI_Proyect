import pandas as pd
import numpy
import numpy as np
import matplotlib.pyplot as plt
import random
#columns = ["DateTime","Temperature","Humidity","Wind Speed","general diffuse flows","diffuse flows","Zone 1 Power Consumption","Zone 2  Power Consumption","Zone 3  Power Consumption"]

df = pd.read_csv('Tetuan City power consumption.csv')
# Convertir la columna 'DateTime' a tipo datetime si no lo has hecho
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extraer el mes y el año de la columna 'DateTime'
df['Month'] = df['DateTime'].dt.month
df['Year'] = df['DateTime'].dt.year

# Agrupar por año y mes, y luego sumar el consumo de energía en la Zona 1 para cada mes
monthly_consumption = df.groupby(['Year', 'Month'])['Zone 1 Power Consumption'].sum().reset_index()

# Ordenar los resultados por consumo de energía de mayor a menor
top_5_months = monthly_consumption.sort_values(by='Zone 1 Power Consumption', ascending=False).head(5)

# Mostrar los 5 meses con mayor consum
print(top_5_months)

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

df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extraer características temporales
df['Hour'] = df['DateTime'].dt.hour
df['Minute'] = df['DateTime'].dt.minute
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# También puedes agregar características cíclicas para la hora
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

# Normalización de las nuevas características temporales
df['Hour_normalized'] = df['Hour'] / 24
df['Minute_normalized'] = df['Minute'] / 60

# Eliminar columnas no necesarias y reordenar
df = df.drop(columns=['Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime', 'Hour', 'Minute'])

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

valor_min = df['Wind Speed'].min()
valor_max = df['Wind Speed'].max()
df['Wind Speed normalized'] = (df['Wind Speed'] - df['Wind Speed'].min()) / (df['Wind Speed'].max() - df['Wind Speed'].min())

df.to_csv('Normalización de datos.csv', index=False)

'''
Editar csv eliminando comolunas no normalizadas
'''

df = df.drop(columns=['Zone 1 Power Consumption', 'Temperature', 'Humidity'])

df = df.round(3)

df.to_csv('Normalización de datos.csv', index=False)

'''
Implementación de la regresión lineal, debido a que mis datos no son clasificados
'''
samples = df[['Hour_normalized', 'Minute_normalized', 'Temperature normalized']].values.tolist()

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
alfa = 0.1
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

#print("Original samples:")
#print(samples)

#print("Original samples:")
#print(samples)
while True:
    oldparams = list(params)
    #print("params")
    params = GD(params, samples, y, alfa)
    show_error(params, samples, y)
    #print("params")
    epochs += 1
    if oldparams == params or epochs == 1000:
        #print("samples:")
        #print(samples)
        print("final params:")
        print(params)
        break


# Visualizar la gráfica de errores
plt.plot(__erros__)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Epochs')
plt.show()


def calculate_r_squared(params, samples, y):
    # Calcular las predicciones
    predictions = [h(params, sample) for sample in samples]
    
    # Calcular el valor promedio de y
    y_mean = np.mean(y)
    
    # Calcular la suma del cuadrado del error de predicción (SSE)
    sse = sum((y_i - pred) ** 2 for y_i, pred in zip(y, predictions))
    
    # Calcular la suma total del cuadrado (SST)
    sst = sum((y_i - y_mean) ** 2 for y_i in y)
    
    # Calcular el R^2
    r_squared = 1 - (sse / sst)
    
    return r_squared

# Calcular el coeficiente de determinación R^2
r_squared_test = calculate_r_squared(params, samples, y)
print(f"R^2 en el conjunto de prueba: {r_squared_test}")
