import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os

# Chargement des données
df = pd.read_csv("./datasets/food.csv")

# Encodage des variables catégorielles



df['Food'] = LabelEncoder().fit_transform(df['food '])  # 0, 1, 2


# Sauvegarder la colonne d'origine du revenu
calories_original = df['Calories '].copy()

# Standardisation séparée pour l'âge et le revenu
scaler_calories = StandardScaler()
scaler = StandardScaler()

df['calories'] = scaler_calories.fit_transform(df[['Calories ']])
df['sugars'] = scaler.fit_transform(df[['Sugars ']])
df['calcium'] = scaler.fit_transform(df[['Calcium ']])

################################### RÉGRESSION PART ###################################

# Variables explicatives et cible
X = df[['Food', 'sugars', 'calcium']]
y = df['calories']  # standardisé

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# ➤ Déstandardisation (revenir aux CFA)
y_pred_real = scaler_calories.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_real = scaler_calories.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

# Enregistrement des prédictions réelles dans un CSV
predictions_df = pd.DataFrame({
    'calories Réel': y_test_real,
    'calories Prédit': y_pred_real
})
predictions_df.to_csv('./predictions_calories.csv', index=False)

# Évaluation sur valeurs réelles
mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_real, y_pred_real)

# Affichage
print("Régression - Prédiction des calory")
print(f"MAE  : {mae:,.0f}")
print(f"MSE  : {mse:,.0f}")
print(f"RMSE : {rmse:,.0f}")
print(f"R²   : {r2:.3f}")

