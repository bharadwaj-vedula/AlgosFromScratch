# %% Imports
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

# %% Dataset Loading
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.15,shuffle = True, random_state= 2025)

# %% Training
print(X_train.shape,y_train.shape)
model = LinearRegression()
model.fit(X_train,y_train)
