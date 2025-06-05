import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
#from tensorflow.keras.utils import to_categorical

# Charger les données
  # Assure-toi que le fichier est dans ton dossier
df = pd.read_csv("./datasets/wines.csv")
# Afficher un aperçu
print(df.head())
print(df.isnull().sum())

# Remplacer les valeurs manquantes (moyenne ou autre stratégie)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convertir la colonne 'type' en numérique (white/red → 0/1)
df['type'] = df['type'].map({'white': 0, 'red': 1})

# Séparer les features et le label
X = df.drop('quality', axis=1)
y = df['quality']

# Afficher les classes possibles
print("Classes de qualité :", sorted(y.unique()))

# Encodage one-hot des labels (classification multiclasse)
# y_cat = to_categorical(y)
y_cat = tf.keras.utils.to_categorical(y)
# Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(y_cat.shape[1], activation='softmax'))  # Sortie = nb de classes (souvent 6 ou 7)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_cat.shape[1], activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    verbose=1)

preds = model.predict(X_test).flatten()

# Métriques
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

import matplotlib.pyplot as plt

# Évaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Courbes d'entraînement
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Performance du modèle')
plt.legend()
plt.show()
