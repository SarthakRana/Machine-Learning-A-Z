#Polynomial Regression

#Importing Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Position_Salaries(Polynomial R).csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting Linear regresssion to the dataset.
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)

#Fitting Polynomial Regression to the dataset.
#poly_reg will transform matrix X to X_poly which will have x1,x1^2,x1^3...
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly, y)

#Visualising the Linear Regression result.
plt.scatter(X, y, color='red')
plt.plot(X, linear_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression result.
#X-grid is for a smoother curve rather than straight lines.
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
linear_reg.predict(6.5)

#Predicting a new result with Polynomial Regression
linear_reg2.predict(poly_reg.fit_transform(6.5))