import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('./dataset/diamond.csv')
# Afficher les 5 premières lignes du DataFrame
print(df.head())  

# sns.boxplot(x="price", y="cut", data=df)
# plt.title("Boxplot des prix selon la vaiable cut")

# 
# average_price = df.groupby(['cut', 'color'])['price'].mean().reset_index()

df_numeric = df.select_dtypes(include=np.number)
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(10, 8))  # Ajustez la taille de la figure si nécessaire
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation des Variables Numériques')

# plt.figure(figsize=(10, 6))
# sns.barplot(x='cut', y='price', hue='color', data=average_price)
# plt.title('Prix moyen par Cut et Couleur')
# plt.xlabel('Cut')
# plt.ylabel('Prix Moyen')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
plt.show()

