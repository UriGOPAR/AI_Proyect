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
    top_2_meses = consumo_por_mes.nlargest(5)

    print(f"Los dos meses con mayor consumo en {zona_max_consumo} son:")
    for mes, consumo in top_2_meses.items():
        print(f"Mes: {mes}, Consumo: {consumo:.2f} unidades")
    return zona_max_consumo, top_2_meses
zona_max_consumo, top_2_meses = zona_mayor_consumo(df)

meses_seleccionados = [8,7,6]
df_seleccionado = df[df['Month'].isin(meses_seleccionados)].reset_index(drop=True)

count = df_seleccionado['Month'].value_counts()
print(count)

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

# Mezclar y dividir el dataset en entrenamiento, validación y prueba
df = df.sample(frac=1).reset_index(drop=True)
train_size = int(0.6 * len(df))
valid_size = int(0.2 * len(df))

train_df = df[:train_size]
valid_df = df[train_size:train_size + valid_size]
test_df = df[train_size + valid_size:]

# Definir características y variable objetivo para cada conjunto
features = ['Hour_normalized','Wind Speed','Temperature','general diffuse flows','diffuse flows','Humidity']

X_train = train_df[features].values
y_train = train_df['Zone 1 normalized'].values

X_valid = valid_df[features].values
y_valid = valid_df['Zone 1 normalized'].values

X_test = test_df[features].values
y_test = test_df['Zone 1 normalized'].values

# Añadir columna de unos (bias) a las muestras estandarizadas
samples_train = np.c_[np.ones((X_train.shape[0], 1)), standardize(X_train)]
samples_valid = np.c_[np.ones((X_valid.shape[0], 1)), standardize(X_valid)]
samples_test = np.c_[np.ones((X_test.shape[0], 1)), standardize(X_test)]

# Función de predicción
def h(params, samples):
    return np.dot(samples, params)

# Función para calcular el error cuadrático medio (MSE)
def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

# Función para calcular R^2
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def GD(params, samples, y, alfa, lambda_reg):
    m = len(samples)
    gradients = [0] * len(params)
    
    for i in range(m):
        error = h(params, samples[i]) - y[i]
        for j in range(len(params)):
            gradients[j] += error * samples[i][j]
    
    for j in range(len(params)):
        if j == 0: 
            gradients[j] = gradients[j] / m
        else:
            gradients[j] = (gradients[j] / m) + (lambda_reg / m) * params[j]
        
        params[j] = params[j] - alfa * gradients[j]
    
    return params

# Variables iniciales
params = np.random.randn(samples_train.shape[1])
lambda_reg = 0.0001
alfa = 0.1
epochs = 1000 

r2_train_list = []
r2_valid_list = []
r2_test_list = []

# Entrenamiento del modelo con regularización L2
for epoch in range(epochs):
    # Actualizar parámetros con descenso de gradiente
    params = GD(params, samples_train, y_train, alfa, lambda_reg)

    # Predicciones en los tres conjuntos
    predictions_train = h(params, samples_train)
    predictions_valid = h(params, samples_valid)
    predictions_test = h(params, samples_test)

    # Calcular R² para cada conjunto y convertir a porcentaje
    r2_train = r_squared(y_train, predictions_train) * 100
    r2_valid = r_squared(y_valid, predictions_valid) * 100
    r2_test = r_squared(y_test, predictions_test) * 100

    # Guardar R²
    r2_train_list.append(r2_train)
    r2_valid_list.append(r2_valid)
    r2_test_list.append(r2_test)

    # Imprimir progreso
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Train R²: {r2_train:.2f}%, Valid R²: {r2_valid:.2f}%, Test R²: {r2_test:.2f}%')

# Graficar la evolución de R² en porcentaje para entrenamiento, validación y prueba
plt.figure(figsize=(10, 6))
plt.plot(r2_train_list, label='Train R²')
plt.plot(r2_valid_list, label='Valid R²', linestyle='--')
plt.plot(r2_test_list, label='Test R²', linestyle=':')
plt.title('Evolución de R² durante el Entrenamiento (en porcentaje)')
plt.xlabel('Época')
plt.ylabel('R² (%)')
plt.legend()
plt.grid(True)
plt.show()
