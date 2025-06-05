import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os

# Chargement des données
df = pd.read_csv("./datasets/diamond.csv")

# Encodage des variables catégorielles
df_drop = df.dropna()
print("\nAprès suppression des lignes avec valeurs manquantes :\n", df_drop)


df['Cut'] = LabelEncoder().fit_transform(df['cut'])  # 0, 1, 2
df['Color'] = LabelEncoder().fit_transform(df['color'])
df['Clarity'] = LabelEncoder().fit_transform(df['clarity'])
# df['Achat'] = l_achats.fit_transform(df['achat'])  # 0: Non, 1: Oui

# Sauvegarder la colonne d'origine du revenu
price_original = df['price'].copy()

# Standardisation séparée pour l'âge et le revenu
scaler_carat = StandardScaler()
scaler_depth = StandardScaler()
scaler_price = StandardScaler()


df['Carat'] = scaler_carat.fit_transform(df[['carat']])
df['Depth'] = scaler_depth.fit_transform(df[['depth']])
df['Price'] = scaler_price.fit_transform(df[['price']])




################################### RÉGRESSION PART ###################################

# Variables explicatives et cible
X = df[['Price', 'Clarity', 'Color', 'Carat', 'Depth']]
y = df['Cut']  # standardisé

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

#➤ Déstandardisation (revenir aux CFA)
# y_pred_real = scaler_price.inverse_transform(y_pred.reshape(-1, 1)).flatten()
# y_test_real = scaler_price.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

# Enregistrement des prédictions réelles dans un CSV
predictions_df = pd.DataFrame({
    'cut Réel': y_test,
    'cut Prédit': y_pred
})
predictions_df.to_csv('./predictions_diamants.csv', index=False)

# # Évaluation sur valeurs réelles
# mae = mean_absolute_error(y_test_real, y_pred_real)
# mse = mean_squared_error(y_test_real, y_pred_real)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test_real, y_pred_real)

# # Affichage
# print("Régression - Prédiction du prix")
# print(f"MAE  : {mae:,.0f}")
# print(f"MSE  : {mse:,.0f}")
# print(f"RMSE : {rmse:,.0f}")
# print(f"R²   : {r2:.3f}")
 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


print("\nClassification - Prédiction du genre")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision : {precision_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
print(f"F1-score  : {f1_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
print("\nRapport de classification :\n", classification_report(y_test, y_pred, zero_division=0))



