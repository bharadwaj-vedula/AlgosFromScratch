# %% Imports
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KMeans import Kmeans

# %% data split
iris = datasets.load_iris()
X, y = iris.data, iris.target

# %% train
clf = Kmeans(k = 3,n_iters=15)
predictions = clf.fit(X)
print(predictions)
# %% test
acc = np.sum(predictions == y) / len(y)
print(acc)
