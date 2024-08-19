import pandas as pd
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
__erros__ = []

def h(params, sample):
    #Calcular la hipotesis
    acum = 0
    for i in range(len(params)):
        acum = acum + params[i]*sample[i]
    return acum

def show_error(params, samples, y):
    #Calcular el error
    global __erros__
    error_acum = 0
    for i in range(len(samples)):
        hyp = h(params, samples[i])
        error = hyp - y[i]
        error_acum = error_acum + error**2
        #Mean error  es el promedio del error para sacar la desviación estandar
        mean_error_param = error_acum / len(samples)
        __erros__.append(mean_error_param)
def GD(params, samples, y, alfa):
    #Gradiente descendiente
    temp = list(params)
    general_error = 0
    for j in range(len(params)):
        acum = 0
        error_acum = 0
        for i in range(len(samples)):
            error = h(params, samples[i]) - y[i]
            acum = acum + error * samples[i][j]
        temp[j] = params[j] - alfa * (1/len(samples)) * acum
    return temp
def scaling(samples):
	acum =0
	samples = numpy.asarray(samples).T.tolist() 
	for i in range(1,len(samples)):	
		for j in range(len(samples[i])):
			acum=+ samples[i][j]
		avg = acum/(len(samples[i]))
		max_val = max(samples[i])
		for j in range(len(samples[i])):
			samples[i][j] = (samples[i][j] - avg)/max_val  #Mean scaling
	return numpy.asarray(samples).T.tolist()

samples = df[['Temperature normalized', 'Humidity normalized', 'Wind Speed', 'general diffuse flows', 'diffuse flows']].values.tolist()
y = df['Zone 1 normalized'].values.tolist()

params = [0, 0, 0, 0, 0, 0]

#tasas de aprendizaje

alfa = 0.0001
for i in range(len(samples)):
	if isinstance(samples[i], list):
		samples[i]=  [1]+samples[i]
	else:
		samples[i]=  [1,samples[i]]
print ("original samples:")
print (samples)
print (samples)

epochs = 0

while True:
    oldparams = list(params)
    print("params")
    params = GD(params, samples, y, alfa)
    show_error(params, samples, y)
    print("params")
    epochs = epochs + 1
    if(oldparams == params or epochs == 2):   #  local minima is found when there is no further improvement
        print ("samples:")
        print(samples)
        print ("final params:")
        print (params)
        break

import matplotlib.pyplot as plt 
# Visualizar la gráfica de errores
plt.plot(__erros__)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Error during Gradient Descent')
plt.show()

