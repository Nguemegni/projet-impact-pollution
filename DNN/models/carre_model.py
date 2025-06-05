import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Chargement
df = pd.read_csv("./data/datasets/carre.csv")
X = df[['x']].values
y = df["y"].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modele
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss="mse")
model.fit(x_train, y_train, epochs=100, verbose=1)

# Evaluation
preds = model.predict(x_test).flatten()

# Métriques
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Matrice de confusion (discrétisation)
# bins = np.linspace(min(y_test.min(), preds.min()), max(y_test.max(), preds.max()), 6)  # 5 classes
# y_test_binned = np.digitize(y_test, bins)
# preds_binned = np.digitize(preds, bins)

# cm = confusion_matrix(y_test_binned, preds_binned)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title("Matrice de confusion (binned)")
# plt.show()

# Scatter plot
plt.scatter(x_test, y_test, label="Vrai")
plt.scatter(x_test, preds, label="Predict", color="red")
plt.legend()
plt.title("Regression y = x^2")
plt.show()