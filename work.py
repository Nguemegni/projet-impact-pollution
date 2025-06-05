import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os

# Chargement des données
df = pd.read_csv("./datasets/work.csv")


# Employee_ID,Employment_Type,Hours_Worked_Per_Week,Productivity_Score,Well_Being_Score


df['employment_type'] = LabelEncoder().fit_transform(df['Employment_Type'])  # 0, 1, 2
# df['Achat'] = l_achats.fit_transform(df['achat'])  # 0: Non, 1: Oui

# Sauvegarder la colonne d'origine du revenu
Productivity_original = df['Productivity_Score'].copy()

# Standardisation séparée pour l'âge et le revenu
scaler = StandardScaler()
scaler_productivity = StandardScaler()

df['hours_worked'] = scaler.fit_transform(df[['Hours_Worked_Per_Week']])
df['productivity'] = scaler_productivity.fit_transform(df[['Productivity_Score']])
df['well_being'] = scaler.fit_transform(df[['Well_Being_Score']])
################################### RÉGRESSION PART ###################################

# Variables explicatives et cible
X = df[['employment_type', 'hours_worked', 'well_being']]
y = df['productivity']  # standardisé

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# ➤ Déstandardisation (revenir aux CFA)
y_pred_real = scaler_productivity.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_real = scaler_productivity.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

# Enregistrement des prédictions réelles dans un CSV
predictions_df = pd.DataFrame({
    'productivity Réel': y_test_real,
    'productivity Prédit': y_pred_real
})
predictions_df.to_csv('./predictions_productivity.csv', index=False)

# Évaluation sur valeurs réelles
mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_real, y_pred_real)

# Affichage
print("Régression - Prédiction du revenu")
print(f"MAE  : {mae:,.0f}")
print(f"MSE  : {mse:,.0f}")
print(f"RMSE : {rmse:,.0f}")
print(f"R²   : {r2:.3f}")

