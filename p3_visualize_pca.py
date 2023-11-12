import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

train_data = pd.read_csv('Reg_Train.txt', sep=' ', header=None)
test_data = pd.read_csv('Reg_Test.txt', sep=' ', header=None)

train_x = np.array(train_data.iloc[:, :-1])
train_y = np.array(train_data.iloc[:, -1])
test_x = np.array(test_data.iloc[:, :-1])
test_y = np.array(test_data.iloc[:, -1])


pca = PCA(n_components=2)

pca.fit(train_x)
w = pca.components_.T
print(w.shape)

tmp = train_x @ w
Xs = tmp[:, 0]
Ys = tmp[:, 1]
Zs = train_y

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xs, Ys, Zs)
plt.show()
plt.close()


tmp = test_x @ w
Xs = tmp[:, 0]
Ys = tmp[:, 1]
Zs = test_y

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xs, Ys, Zs)
plt.show()
plt.close()
