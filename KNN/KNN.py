# %% Imports
import numpy as np
from collections import Counter

# %% Cell 1
def eucledian_distance(list1:list[int],list2:list[int]) -> int:
    assert len(list1) == len(list2)
    res:int = 0
    for i in range(0,len(list2)):
        res += (list1[i]-list2[i])**2


    return res**(1/2)

# %% Cell 2
class KNN:
    def __init__(self,k:int = 3):
        self.k = k

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X_test):
        #for each point in x test calculate euclidean distance
        preds = []
        for p in X_test:
            distances = [eucledian_distance(p,train_point) for train_point in self.X]
            k_inidices = np.argsort(distances)[:self.k]
            k_nearest_values = [self.y[i] for i in k_inidices]

            pred = Counter(k_nearest_values).most_common()[0][0]
            preds.append(pred)

        return preds
