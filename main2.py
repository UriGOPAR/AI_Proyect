import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Tetuan City power consumption.csv')

# Convertir 'DateTime' a formato de fecha y hora
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extraer características temporales
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


# Filtrar las filas que tienen el mes 8
df_august = df[df['Month'] == 8]

# Si deseas eliminar las demás filas del dataframe original
df = df_august.reset_index(drop=True)

print(df.head())

# Dividir el dataset en entrenamiento y prueba
# train_size = int(0.8 * len(df))
# train_df = df[:train_size]  # 80% para entrenamiento
# test_df = df[train_size:]   # 20% para prueba

df = df.sample(frac=1).reset_index(drop=True)
train_df = df[:int(0.8 * len(df))]
test_df = df.drop(train_df.index)


# Separar las variables independientes (X) y la variable dependiente (y)
features = ['Hour','Minute','Day','Hour_sin',"Humidity", "Wind Speed", "general diffuse flows", "diffuse flows"]
X_train = train_df[features].values
y_train = train_df['Zone 1 Power Consumption'].values

X_test = test_df[features].values
y_test = test_df['Zone 1 Power Consumption'].values

# Calcular la media y desviación estándar de las características en el conjunto de entrenamiento
means = np.mean(X_train, axis=0)
stds = np.std(X_train, axis=0)

# Estandarizar las características (tanto en entrenamiento como en prueba)
X_train_standardized = (X_train - means) / stds
X_test_standardized = (X_test - means) / stds

# Estandarizar la variable dependiente también
y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)
y_train_standardized = (y_train - y_train_mean) / y_train_std
y_test_standardized = (y_test - y_train_mean) / y_train_std

# Añadir columna de unos (bias) a las muestras estandarizadas
samples_train = np.c_[np.ones(X_train_standardized.shape[0]), X_train_standardized]
samples_test = np.c_[np.ones(X_test_standardized.shape[0]), X_test_standardized]

# Inicializar los parámetros
params = np.random.randn(samples_train.shape[1]) * .1
alfa = 0.1
lambda_reg = 0.01  # Término de regularización L2
epochs = 0
__erros__ = []

def h(params, sample):
    return np.dot(params, sample)

def show_error(params, samples, y, lambda_reg):
    global __erros__
    errors = np.dot(samples, params) - y
    mean_error_param = np.mean(errors**2) / 2
    # Agregar el término de regularización L2 al error
    regularization_term = (lambda_reg / 2) * np.sum(params[1:]**2)
    total_error = mean_error_param + regularization_term
    __erros__.append(total_error)
    return total_error

def GD(params, samples, y, alfa, lambda_reg):
    m = len(samples)
    errors = np.dot(samples, params) - y
    gradient = np.dot(samples.T, errors) / m
    # Agregar el término de regularización L2 al gradiente
    gradient[1:] += (lambda_reg / m) * params[1:]
    params = params - alfa * gradient
    return params

# Entrenamiento del modelo con L2 Regularization
while True:
    oldparams = params.copy()
    params = GD(params, samples_train, y_train, alfa, lambda_reg)
    error = show_error(params, samples_train, y_train, lambda_reg)
    print(f"Epoch {epochs}, Error: {error}")
    epochs += 1
    if np.allclose(oldparams, params) or epochs == 1000:
        print("Final params:")
        print(params)
        break

# Calcular R² en el conjunto de prueba
def calculate_r_squared(params, samples, y):
    predictions = np.dot(samples, params)
    sse = np.sum((y - predictions) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (sse / sst)
    return r_squared

r_squared_test = calculate_r_squared(params, samples_test, y_test)
print(f"R² en el conjunto de prueba: {r_squared_test}")

# Graficar el error durante las épocas
plt.plot(__erros__)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Epochs')
plt.show()