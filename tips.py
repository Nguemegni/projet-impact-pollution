import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.lmplot(x="total_bill", y="tip", hue="sex", col="smoker", data=df)
plt.subplots_adjust(top=0.9)
plt.suptitle('Relation entre la facture totale et le pourboire, segmentée par sexe et fumeur')
plt.tight_layout()
sns.boxenplot(x="day", y="tip", hue="sex", data=df)
plt.title('Distribution des Pourboires par Jour et Sexe')


g = sns.FacetGrid(df, col="smoker", row="time")
g.map(sns.histplot, "tip")
g.map(sns.histplot, "tip")
g.set_axis_labels("Pourboire", "Fréquence")
g.set_titles(col_template="{col_name} Fumeur", row_template="{row_name}")
plt.show()
