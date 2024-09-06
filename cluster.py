import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Filtrar los meses de julio y agosto
meses_seleccionados = [7, 8]
df = df[df['Month'].isin(meses_seleccionados)].reset_index(drop=True)

# Eliminar columnas innecesarias
df = df.drop(columns=['Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime'])

# Preparación de los datos
features = ['Hour_normalized', 'Wind Speed', 'Temperature', 'general diffuse flows', 'diffuse flows', 'Humidity']

# Aplicar K-Means con 3 clústeres
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[features])

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df[features])

# Comparar visualmente los clústeres generados por K-Means y DBSCAN
feature_1 = 'Temperature'
feature_2 = 'Zone 1 normalized'

plt.figure(figsize=(10, 6))
plt.scatter(df[feature_1], df[feature_2], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Cluster K-Means')
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title('Visualización de los Clústeres de K-Means')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df[feature_1], df[feature_2], c=df['DBSCAN_Cluster'], cmap='plasma', alpha=0.6)
plt.colorbar(label='Cluster DBSCAN')
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title('Visualización de los Clústeres de DBSCAN')
plt.show()

# Evaluación del coeficiente de silueta para ambos métodos
silhouette_kmeans = silhouette_score(df[features], df['Cluster'])
print(f"Coeficiente de Silueta para K-Means: {silhouette_kmeans:.4f}")

mask = df['DBSCAN_Cluster'] != -1  # Ignorar el ruido en DBSCAN
silhouette_dbscan = silhouette_score(df[features][mask], df['DBSCAN_Cluster'][mask])
print(f"Coeficiente de Silueta para DBSCAN: {silhouette_dbscan:.4f}")

# Visualización de la inercia para K-Means
range_n_clusters = range(2, 11)
inertias = []
silhouette_coefficients = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df[features])
    
    inertias.append(kmeans.inertia_)
    score = silhouette_score(df[features], kmeans.labels_)
    silhouette_coefficients.append(score)

plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, inertias, marker='o')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.title('Inercia vs Número de Clústeres')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_coefficients, marker='o', color='orange')
plt.xlabel('Número de Clústeres')
plt.ylabel('Coeficiente de Silueta')
plt.title('Silhouette vs Número de Clústeres')
plt.show()

# Visualización de los clústeres en 2D
plt.figure(figsize=(10, 6))
plt.scatter(df[feature_1], df[feature_2], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Cluster K-Means')
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title('Visualización de los Clústeres de K-Means')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df[feature_1], df[feature_2], c=df['DBSCAN_Cluster'], cmap='plasma', alpha=0.6)
plt.colorbar(label='Cluster DBSCAN')
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title('Visualización de los Clústeres de DBSCAN')
plt.show()
