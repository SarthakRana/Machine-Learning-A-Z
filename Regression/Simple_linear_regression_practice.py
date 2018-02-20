#Simple Linear Regression

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Splitting dataset into training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Fitting Simple linear regression to Training set
"""
Here we are making our machine learn about the training set
and hence draw a line/hypothesis which is the best fit line or
the best predictor.  
"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) 

#Predict the Test set results
#We will use hypothesis here to predict results for test set.
#Y_pred is a vector of predicted salaries
Y_pred = regressor.predict(X_test) 

#Visualising the Training set results
#red color for real values.
#blue color for predicted values and regression line.
plt.scatter(X_train, Y_train, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary v/s Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary v/s Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()