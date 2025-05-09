import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler,Binarizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df= pd.read_csv("./datasets/processing.csv")

print("Données originales :\n", df)

print("\nValeurs manquantes par colonne :\n", df.isna().sum())

# df_drop = df.dropna()
# print("\nAprès suppression des lignes avec valeurs manquantes :\n", df_drop)


encoder = LabelEncoder()
df['genre'] = encoder.fit_transform(df['genre'])
df['achats'] = encoder.fit_transform(df['achats'])
df['profession'] = encoder.fit_transform(df['profession'])

scaler = StandardScaler()
df[['age','revenu']] = scaler.fit_transform(df[['age','revenu']])
print("Données standardiser :\n", df[['age','revenu']])



df = df.drop(columns=['age'])
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',fmt=".2f", linewidths=.5)
plt.title("correlation map")
plt.show()


binarizer = Binarizer(threshold=0)
df['binaire'] = binarizer.fit_transform(df[['revenu']])
print("Données binariser :\n", df[['binaire']])

df.to_csv('./datasets/newprocessed.csv')

