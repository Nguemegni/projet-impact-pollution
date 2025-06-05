import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler,Binarizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df= pd.read_csv("./datasets/heart.csv")

# print("Données originales :\n", df)

# print("\nValeurs manquantes par colonne :\n", df.isna().sum())

# df_drop = df.dropna()
# print("\nAprès suppression des lignes avec valeurs manquantes :\n", df_drop)
#gender,age,height_cm,weight_kg,duration_min,calories_burned,satisfaction,membership_type

# encoder = LabelEncoder()
# df['Gneder'] = encoder.fit_transform(df['gender'])
# df['Membership_type'] = encoder.fit_transform(df['membership_type'])

# scaler = StandardScaler()
# df['Age'] = scaler.fit_transform(df[['age']])
# df['Calories_burned'] = scaler.fit_transform(df[['calories_burned']])
# df['Satisfaction'] = scaler.fit_transform(df[['satisfaction']])
# df['Duration_min'] = scaler.fit_transform(df[['duration_min']])
# df['Height_cm'] = scaler.fit_transform(df[['height_cm']])
# df['Weight_kg'] = scaler.fit_transform(df[['weight_kg']])

# # print("Données standardiser :\n", df[[[[[['Age','Pregnancies','Glucose','Insulin','BMI','SkinThickness','DiabetesPedigreeFunction','BloodPressure']]]]]])



# df = df.drop(columns=['age'])
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',fmt=".2f", linewidths=.5)
plt.title("correlation map")
plt.show()


# binarizer = Binarizer(threshold=0)
# df['binaire'] = binarizer.fit_transform(df[['revenu']])
# print("Données binariser :\n", df[['binaire']])

# df.to_csv('./dataset/newdiab.csv')

