import pandas as pd

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

