#Multiple Linear Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('50_Startups(Multiple LR).csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
#Linear Regression library does take care of this trap
#Here we've removed the first column.
X = X[:, 1:]

#Splitting dataset into Training set and Test set.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y ,test_size=0.2, random_state=0)

#Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
#statsmodels library does not take into account constant part of regression equation.
#linear_model library includes the constant part but this does not.
#Here we are gonna add a column of Xo=1 for constant part.
# OLS-Ordinary Least Squares
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
"""X_opt array is a team of independent variables that is statistically 
significant in obtaining results or that has major impact on predicted results""" 
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()
X_optimal = X[:, [0, 1, 2, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()
X_optimal = X[:, [0, 1, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()
X_optimal = X[:, [0, 1, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()