import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Analizar y procesar un dataset de consumo de energía eléctrica en una ciudad
Conocer la información general del dataset, verificar duplicados y filas vacías
Convertir la columna 'DateTime' a formato de fecha y hora
'''
df = pd.read_csv('Tetuan City power consumption.csv')


print(df.head())
df.info()

duplicate = df.duplicated()
filas_duplicadas = df[duplicate]
print(filas_duplicadas)

vacio = df.isnull().sum()
print(vacio[vacio > 0])

filas_vacias = df[df.isnull().any(axis=1)]
print(filas_vacias)

df['DateTime'] = pd.to_datetime(df['DateTime'])

df['Hour'] = df['DateTime'].dt.hour
df['Minute'] = df['DateTime'].dt.minute
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# Agregar características cíclicas para la hora
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

# Normalización de características temporales
df['Hour_normalized'] = df['Hour'] / 24
df['Minute_normalized'] = df['Minute'] / 60



def zona_mayor_consumo(df):
    #Poder saber la zona con mayor consumo total de energía y los dos meses con mayor consumo en esa zona
    consumo_zona_1 = df['Zone 1 Power Consumption'].sum()
    consumo_zona_2 = df['Zone 2  Power Consumption'].sum()
    consumo_zona_3 = df['Zone 3  Power Consumption'].sum()

    consumos = {'Zone 1': consumo_zona_1, 'Zone 2': consumo_zona_2, 'Zone 3': consumo_zona_3}
    zona_max_consumo = max(consumos, key=consumos.get)
    
    print(f"La zona con mayor consumo total de energía es: {zona_max_consumo} con un consumo de {consumos[zona_max_consumo]:.2f} unidades.")
    columna_zona = f"{zona_max_consumo} Power Consumption"
    consumo_por_mes = df.groupby('Month')[columna_zona].sum()
    top_2_meses = consumo_por_mes.nlargest(2)

    print(f"Los dos meses con mayor consumo en {zona_max_consumo} son:")
    for mes, consumo in top_2_meses.items():
        print(f"Mes: {mes}, Consumo: {consumo:.2f} unidades")
    return zona_max_consumo, top_2_meses
zona_max_consumo, top_2_meses = zona_mayor_consumo(df)



'''
Filtrar los meses por el mes de agosto 
Eliminar las columnas 'Zone 2 Power Consumption', 'Zone 3 Power Consumption' y 'DateTime'
Normalizar las características 'Zone 1 Power Consumption', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows' y 'diffuse flows'
'''
#df_august = df[df['Month'] == 8]
#df = df_august.reset_index(drop=True)

meses_seleccionados = [8, 7]
df_seleccionado = df[df['Month'].isin(meses_seleccionados)].reset_index(drop=True)


# Normalización de 'Zone 1 Power Consumption'
valor_min = df['Zone 1 Power Consumption'].min()
valor_max = df['Zone 1 Power Consumption'].max()
df['Zone 1 normalized'] = (df['Zone 1 Power Consumption'] - valor_min) / (valor_max - valor_min)

# Normalización de 'Temperature'
valor_min = df['Temperature'].min()
valor_max = df['Temperature'].max()
df['Temperature normalized'] = (df['Temperature'] - valor_min) / (valor_max - valor_min)

# Normalización de 'Humidity'
valor_min = df['Humidity'].min()
valor_max = df['Humidity'].max()
df['Humidity normalized'] = (df['Humidity'] - valor_min) / (valor_max - valor_min)

# Normalización de 'Wind Speed'
valor_min = df['Wind Speed'].min()
valor_max = df['Wind Speed'].max()
df['Wind Speed normalized'] = (df['Wind Speed'] - valor_min) / (valor_max - valor_min)

# Normalización de 'general diffuse flows'
valor_min = df['general diffuse flows'].min()
valor_max = df['general diffuse flows'].max()
df['general diffuse flows normalized'] = (df['general diffuse flows'] - valor_min) / (valor_max - valor_min)

# Normalización de 'diffuse flows'
valor_min = df['diffuse flows'].min()
valor_max = df['diffuse flows'].max()
df['diffuse flows normalized'] = (df['diffuse flows'] - valor_min) / (valor_max - valor_min)


df = df.drop(columns=['Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime'])

# Aplicar la función de escalado a las características normalizadas
def standardize(samples):
    samples_transposed = list(zip(*samples))
    standardized_samples = []
    for feature in samples_transposed:
        mean = np.mean(feature)
        std = np.std(feature)
        standardized_feature = [(x - mean) / std for x in feature]
        standardized_samples.append(standardized_feature)
    standardized_samples = list(zip(*standardized_samples))
    standardized_samples = [list(sample) for sample in standardized_samples]
    return standardized_samples

# Seleccionar las columnas normalizadas para la matriz de correlación
normalized_features = [
    'Zone 1 normalized', 
    'Temperature normalized', 
    'Humidity normalized', 
    'Wind Speed normalized', 
    'general diffuse flows normalized', 
    'diffuse flows normalized',
    'Hour_normalized',
    'Minute_normalized',
]

# Escalar las características normalizadas
standardized_df = df[normalized_features].values
standardized_df = standardize(standardized_df)
df_standardized = pd.DataFrame(standardized_df, columns=normalized_features)

# Calcular y visualizar la matriz de correlación
corr_matrix = df[normalized_features].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación de Variables Normalizadas')
plt.show()


for feature in normalized_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[feature], y=df['Zone 1 normalized'])
    plt.title(f'{feature} vs Zone 1 Power Consumption')
    plt.xlabel(feature)
    plt.ylabel('Zone 1 Power Consumption')
    plt.show()

# Mezclar y dividir el dataset en entrenamiento y prueba
df = df.sample(frac=1).reset_index(drop=True)
train_df = df[:int(0.8 * len(df))]
test_df = df.drop(train_df.index)

features = ['Hour_normalized','Hour_sin','Wind Speed normalized','Temperature normalized', 'general diffuse flows normalized']
X_train = train_df[features].values
y_train = train_df['Zone 1 normalized'].values

X_test = test_df[features].values
y_test = test_df['Zone 1 normalized'].values

# Añadir columna de unos (bias) a las muestras estandarizadas
samples_train = np.c_[np.ones(X_train.shape[0]), standardize(X_train)]
samples_test = np.c_[np.ones(X_test.shape[0]), standardize(X_test)]


params = np.random.randn(samples_train.shape[1]) 
alfa = 0.1
epochs = 0
__erros__ = []


def h(params, sample):
    hypothesis = 0
    for i in range(len(params)):
        hypothesis += params[i] * sample[i]
    return hypothesis

# Función para mostrar el error cuadrático medio (MSE)
def show_error(params, samples, y):
    global __erros__
    total_error = 0
    m = len(samples)
    
    for i in range(m):
        error = h(params, samples[i]) - y[i]
        total_error += error**2
    
    mean_error_param = total_error / (2 * m)
    __erros__.append(mean_error_param)
    return mean_error_param

# Función de descenso de gradiente para actualizar los parámetros θ usando ciclos for
def GD(params, samples, y, alfa):
    m = len(samples)
    gradients = [0] * len(params)
    
    for i in range(m):
        error = h(params, samples[i]) - y[i]
        for j in range(len(params)):
            gradients[j] += error * samples[i][j]
    
    for j in range(len(params)):
        gradients[j] = gradients[j] / m
        params[j] = params[j] - alfa * gradients[j]
    
    return params


# Entrenamiento del modelo
while True:
    oldparams = params.copy()
    params = GD(params, samples_train, y_train, alfa)
    error = show_error(params, samples_train, y_train)
    print(f"Epoch {epochs}, Error: {error}")
    epochs += 1
    if np.allclose(oldparams, params) or epochs == 10000:
        print("Final params:")
        print(params)
        break

def r_squared(y_true, y_pred):
    # Calcular R^2 para saber qué tan bien se ajusta el modelo a los datos
    ss_res = np.sum((y_true - y_pred) ** 2)  
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  
    return r2

predictions = np.dot(samples_test, params)

r2_test = r_squared(y_test, predictions)
print(f"R^2 en el conjunto de prueba: {r2_test:.4f}")


plt.plot(__erros__)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Epochs')
plt.show()


plt.figure(figsize=(10, 6))

plt.scatter(range(len(y_test)), y_test, color='green', label='Valores Reales', alpha=0.6)
plt.scatter(range(len(predictions)), predictions, color='blue', label='Predicciones', alpha=0.6)
plt.xlabel('Índice')
plt.ylabel('Consumo de Energía Normalizado')
plt.title('Valores Reales vs Predicciones')
plt.legend()
plt.show()