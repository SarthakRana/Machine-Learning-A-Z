#SVR

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Position_Salaries(SVR).csv')
X = dataset.iloc[:,1:2]
y = dataset.iloc[:, 2]

#Feature Scaling
#Most classes have feature scaling property of their own.
#SVR class does not has feature scaling property so we need to do here.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
 
#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, y)

#Predicting a new result
"""
Value for CEO cannot be predicted as it is consider as an outlier
because the model takes into account some penalty parameters by default 
and the value of CEO is very far according to the parameters from the 
model therefore no prediction 
"""
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising the SVR results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid =X_grid.reshape((len(X_grid),1))
plt.scatter(X, y ,color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()