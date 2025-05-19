# %% Imports
import numpy as np
from turtle import Shape

# %% KMeans class
class Kmeans:
    def __init__(self,k:int = 3 ,n_iters:int = 100):
        self.k = k
        self.n_iters = n_iters

    def fit(self,X_train):
        X_train = np.array(X_train)
        self.n_samples, self.n_features = X_train.shape

        # Initialize random centroids
        random_idx = np.random.choice(self.n_samples,self.k,replace = False)
        self.centroids = [[X_train[idx]] for idx in random_idx]

        # calculate the eucledian distance
        for _ in range(self.n_iters):
            if _ % 10 == 0:
                print(f'Iteration: {_} Centroid : {self.centroids}')
            self._assign_cluster(X_train)
            self._update_centroids(X_train,self.clusters)


        y = self._get_labels(self.clusters)
        return y

    def _euclidean_distance(self,value1,value2) -> np.float64 :
        dist = np.sqrt(np.sum((value1-value2)**2))
        return dist

    def _assign_cluster(self,X_train):
        self.clusters = [[] for _ in range(self.k)]
        for idx,ele in enumerate(X_train):
            dist = [self._euclidean_distance(ele,centroid) for centroid in self.centroids]
            cluster = np.argmin(dist)
            self.clusters[cluster].append(idx)


    def _update_centroids(self,X_train,clusters):
        for i in range(self.k):
            self.centroids[i] = np.mean(X_train[clusters[i]],axis = 0)

    def _get_labels(self,clusters):
        labels:list[int] = []
        mapping = dict()
        for i in range(0,self.k):
            for idx in clusters[i]:
                mapping[idx] = i

        for j in range(self.n_samples):
            labels.append(mapping[j])
        
        return labels





# %% Testing
a = np.array([[1,2],[3,5],[5,6]])
a.mean(axis= 0)
