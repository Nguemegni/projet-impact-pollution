import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset/titanic.csv')
# sns.countplot(hue="Sex", x="Pclass_1", data=df)
# sns.violinplot(x="Fare", y="Pclass_2", data=df)
df_numeric = df.select_dtypes(include=np.number)
correlation_matrix = df_numeric.corr() 
plt.figure(figsize=(10, 8))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation des Variables Numériques')
plt.show()
# plt.title("countplot des survivants")
# plt.show()