import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os

# Chargement des données
df = pd.read_csv("./datasets/heart.csv")

df['sex'] = LabelEncoder().fit_transform(df['Sex'])  # 0, 1, 2
df['chestpaintype'] = LabelEncoder().fit_transform(df['ChestPainType'])


scaler_Age = StandardScaler()
scaler_Cholesterol = StandardScaler() 
scaler_Heart = StandardScaler()

df['age'] = scaler_Age.fit_transform(df[['Age']])
df['cholesterol'] = scaler_Cholesterol.fit_transform(df[['Cholesterol']])
df['Heart'] = scaler_Heart.fit_transform(df[['HeartDisease']])


################################### RÉGRESSION PART ###################################

# Variables explicatives et cible
X = df[['age', 'cholesterol', 'sex', 'chestpaintype']]
y = df['Heart']  # standardisé

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# ➤ Déstandardisation (revenir aux CFA)
y_pred_real = scaler_Heart.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_real = scaler_Heart.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

# Enregistrement des prédictions réelles dans un CSV
predictions_df = pd.DataFrame({
    'heartdisease Réel': y_test_real,
    'heartdisease Prédit': y_pred_real
})
predictions_df.to_csv('./predictions_heartdisease.csv', index=False)

# Évaluation sur valeurs réelles
mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_real, y_pred_real)

# Affichage
print("Régression - Prédiction du heatdisease")
print(f"MAE  : {mae:,.0f}")
print(f"MSE  : {mse:,.0f}")
print(f"RMSE : {rmse:,.0f}")
print(f"R²   : {r2:.3f}")

