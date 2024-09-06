import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Cargar el dataset
df = pd.read_csv('Tetuan City power consumption.csv')

# Convertir la columna 'DateTime' a formato de fecha y hora
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

# Normalización de 'Zone 1 Power Consumption'
df['Zone 1 normalized'] = (df['Zone 1 Power Consumption'] - df['Zone 1 Power Consumption'].min()) / \
                           (df['Zone 1 Power Consumption'].max() - df['Zone 1 Power Consumption'].min())


meses_seleccionados = [7, 8,6]
df = df[df['Month'].isin(meses_seleccionados)].reset_index(drop=True)

# Eliminar columnas innecesarias
df = df.drop(columns=['Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime'])

# Preparación de los datos
features = ['Hour_normalized', 'Wind Speed', 'Temperature', 'general diffuse flows', 'diffuse flows', 'Humidity']
X = df[features].values
y = df['Zone 1 normalized'].values

# Mezclar el dataset antes de dividirlo
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# División de los datos en 60% entrenamiento, 20% validación y 20% prueba
train_size = 0.6
validation_size = 0.2
test_size = 0.2

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (validation_size + test_size)), random_state=42)

# Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convertir a tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# --- ENTRENAMIENTO DEL MODELO RANDOM FOREST SIMULANDO ÉPOCAS ---
n_estimators = np.arange(10, 200, 10)  # Incrementar en 10 para mayor resolución
train_r2_scores_rf = []
val_r2_scores_rf = []
test_r2_scores_rf = []

# Entrenar el modelo Random Forest con diferentes cantidades de árboles
for n in n_estimators:
    rf_model = RandomForestRegressor(n_estimators=n, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predecir en el conjunto de entrenamiento, validación y prueba
    rf_predictions_train = rf_model.predict(X_train)
    rf_predictions_val = rf_model.predict(X_val)
    rf_predictions_test = rf_model.predict(X_test)

    # Calcular R² para cada conjunto
    train_r2 = r2_score(y_train, rf_predictions_train)
    val_r2 = r2_score(y_val, rf_predictions_val)
    test_r2 = r2_score(y_test, rf_predictions_test)

    # Guardar los R²
    train_r2_scores_rf.append(train_r2)
    val_r2_scores_rf.append(val_r2)
    test_r2_scores_rf.append(test_r2)

    # Imprimir los resultados por cada número de árboles
    print(f'Número de árboles: {n}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}')

# Gráfica de la evolución de R² para Random Forest simulando épocas
plt.figure(figsize=(10, 6))
plt.plot(n_estimators, train_r2_scores_rf, label='R² Entrenamiento (RF)', color='green')
plt.plot(n_estimators, val_r2_scores_rf, label='R² Validación (RF)', color='blue')
plt.plot(n_estimators, test_r2_scores_rf, label='R² Prueba (RF)', linestyle='--', color='red')
plt.xlabel('Número de Árboles (n_estimators)')
plt.ylabel('R²')
plt.title('Evolución del R² para Random Forest (Simulando Épocas)')
plt.legend()
plt.grid(True)
plt.show()

# --- ENTRENAMIENTO DEL MODELO DE RED NEURONAL ---
class EnergyConsumptionModel(nn.Module):
    def __init__(self):
        super(EnergyConsumptionModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128) 
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 32)  
        self.fc4 = nn.Linear(32, 1)   

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instanciar el modelo
model = EnergyConsumptionModel()

# Definir criterio de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_losses = []
train_r2_scores_nn = [] 
val_r2_scores_nn = []
test_r2_scores_nn = []

# Función para calcular R²
def r_squared(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2.item()

# Entrenamiento de la red neuronal
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass (entrenamiento)
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass y optimización
    loss.backward()
    optimizer.step()

    # Evaluación en conjunto de validación y prueba
    model.eval()
    with torch.no_grad():
        train_r2 = r_squared(y_train_tensor, outputs)  # Calcular el R^2 del conjunto de entrenamiento
        train_r2_scores_nn.append(train_r2)

        val_outputs = model(X_val_tensor)
        val_r2 = r_squared(y_val_tensor, val_outputs)
        val_r2_scores_nn.append(val_r2)

        test_outputs = model(X_test_tensor)
        test_r2 = r_squared(y_test_tensor, test_outputs)
        test_r2_scores_nn.append(test_r2)

    # Guardar el loss de entrenamiento
    train_losses.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}')

# Gráfica de la evolución de R² para Red Neuronal
plt.figure(figsize=(10, 6))
plt.plot(train_r2_scores_nn, label='R² Entrenamiento (NN)', color='green')
plt.plot(val_r2_scores_nn, label='R² Validación (NN)', color='blue')
plt.plot(test_r2_scores_nn, label='R² Prueba (NN)', linestyle='--', color='red')
plt.xlabel('Épocas')
plt.ylabel('R²')
plt.title('Evolución del R² durante el entrenamiento (Red Neuronal)')
plt.legend()
plt.grid(True)
plt.show()

# Comparación del R² en validación y prueba para ambos modelos
modelos = ['RF - Val', 'RF - Test', 'NN - Val', 'NN - Test']
r2_values = [val_r2, test_r2, val_r2_scores_nn[-1], test_r2_scores_nn[-1]]

plt.figure(figsize=(10, 6))
plt.bar(modelos, r2_values, color=['orange', 'red', 'blue', 'purple'])
plt.ylabel('R²')
plt.title('Comparación del R² entre Validación y Prueba para Random Forest y Red Neuronal')
plt.show()
