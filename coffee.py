import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os

# Chargement des données
df = pd.read_csv("./datasets/coffee.csv")
#date,datetime,cash_type,money,coffee_name
label = LabelEncoder()
label_cash = LabelEncoder()

df['coffee'] = label.fit_transform(df['coffee_name'])  # 0, 1, 2 selon les genres distincts
df['Cash'] = label_cash.fit_transform(df['cash_type'])
df['Datetime'] = label_cash.fit_transform(df['datetime'])

# Standardisation des variables numériques
scaler = StandardScaler()
df[['Money']] = scaler.fit_transform(df[['money']])

################################### Classification PART ###################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Variables explicatives et cible
X = df[['Datetime','Money','coffee']]  # on ne met pas "Genre" ici car c’est la cible
y = df['Cash']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Enregistrement des prédictions dans un fichier CSV
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('./predictions_classification.csv', index=False)

# Évaluation
print("\nClassification - Prédiction du cashtype")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision : {precision_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
print(f"F1-score  : {f1_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
print("\nRapport de classification :\n", classification_report(y_test, y_pred, zero_division=0))

