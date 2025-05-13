#%% Imports
import numpy as np

# %% LR class
# Steps
# 1. initialize random W & b
# 2. get y pred with those values and get loss function value
# 3. do gradient descent and update weights
#   dj/dw = 1/N(Σ(2x(y_pred-y)))
#   db/dw = 1/N(Σ(2(y_pred-y)))
# 4. repeat n times
class LinearRegression:
    def __init__(self,lr:float = 1e-3,n_iters:int = 1_000):
        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None

    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        n_samples,n_features = X.shape
        self.W = np.random.rand(n_features)
        self.b = np.random.rand(1)
        print(self.b.shape)
        for _ in range(0,self.n_iters):
            y_pred = np.dot(X, self.W) + self.b
            print(y_pred)
            print(self.b)
            dw = (1/n_samples)* (2*np.dot(X.T,(y_pred-y)))
            db = (1/n_samples) * (y_pred-y)

            self.W -= (self.lr*dw)
            self.b -= (self.lr*db)
        print(self.W.shape,self.b.shape)
    def predict(self,X_test):
        X_test = np.array(X_test)
        y_pred = np.dot(X_test,self.W)+ self.b

        return y_pred

# %% testing
model = LinearRegression(n_iters=10)
X = np.array([[1,2,3,4],[5,6,7,8]])
y = np.array([16,35])
model.fit(X,y)
print(model.predict([1,2,3,4]))
np.random.rand(1)
