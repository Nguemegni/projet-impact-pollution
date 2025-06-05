import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Dossier de sortie
output_dir = "datasets/"

# 1. y = x^2
x = np.random.uniform(-2, 2, size=(800, 1))
y = x**2
df = pd.DataFrame(np.hstack([x, y]), columns=["x", "y"])
df.to_csv(output_dir + "carre.csv", index=False)

# 2. XOR (800 répétitions)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
X_rep = np.tile(X, (200,1))
y_rep = np.tile(y, 200)
df = pd.DataFrame(np.hstack([X_rep, y_rep.reshape(-1,1)]), columns=["x1", "x2", "label"])
df.to_csv(output_dir + "xor.csv", index=False)

# 3. AND logique
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
X_rep = np.tile(X, (200,1))
y_rep = np.tile(y, 200)
df = pd.DataFrame(np.hstack([X_rep, y_rep.reshape(-1,1)]), columns=["x1", "x2", "label"])
df.to_csv(output_dir + "and.csv", index=False)

# 4. Cercle (classification binaire)
X = np.random.uniform(-1.5, 1.5, size=(800, 2))
r = 1
y = np.where(X[:,0]**2 + X[:,1]**2 <= r**2, 1, 0)
df = pd.DataFrame(np.hstack([X, y.reshape(-1,1)]), columns=["x", "y", "label"])
df.to_csv(output_dir + "cercle.csv", index=False)

# 5. IRIS dataset
iris = load_iris()
df = pd.DataFrame(data=np.hstack([iris.data, iris.target.reshape(-1,1)]),
                  columns=iris.feature_names + ['label'])
# Étendre à 800 lignes (réplication)
df = pd.concat([df]*6).reset_index(drop=True).iloc[:800]
df.to_csv(output_dir + "iris.csv", index=False)

print("Tous les datasets ont été générés dans le dossier 'data/'")
