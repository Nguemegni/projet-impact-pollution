import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# Charger les données
df = pd.read_csv("../data/datasets/xor2.csv")
X = df[["x1", "x2"]].values
y = df["label"].values

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=50, verbose=0)

# Évaluer les performances
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Matrice de confusion
conf_mat = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :\n", conf_mat)

# Affichage graphique
plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.title("Matrice de confusion - XOR")
plt.tight_layout()
plt.show()

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
