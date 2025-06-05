import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("./data/datasets/and2.csv")
X = df[["x1", "x2"]].values
y = df["label"].values

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle (peut être un perceptron simple)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# regression ==> loss = MSE (Mean Squared Error)
# classification binaire ==> binary_crossentropy
# classification multi classe ==> categorial_crossentropy



# Entraînement
model.fit(X_train, y_train, epochs=50, verbose=2)

# Prédictions
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
plt.title("Matrice de confusion - AND")
plt.tight_layout()
plt.show()

# Rapport
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
