import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset/penguins.csv')
# sns.pairplot(df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'species']], hue="species")
# plt.title("Relation entre la longueur et la largeur du bec par espèce et par l'espece")
# sns.boxplot(x="body_mass_g", y="island", data=df)
# plt.title("Boxplot de comparaison entre mass corporelle et ile")
# sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='species', style='sex')
# plt.title('Relation entre la longueur et la largeur du bec par espèce et par sexe')
# plt.xlabel('Longueur du bec (mm)')
# plt.ylabel('Largeur du bec (mm)')
# plt.legend(title='Espèce et Sexe')
# plt.show()


pivot_table = pd.pivot_table(df, values='bill_length_mm', index='species', columns='bill_depth_mm')

print(pivot_table)
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True)
plt.show()