#Dataset Preprocessing

#Importing Libraries
import pandas as pd   # For importing datasets and managing datasets
import matplotlib.pyplot as plt
import numpy as np   # contains mathematical tools which helps in calculations


#Importing the Dataset
#csv- comma separated values
#tsv- tab separated values
dataset = pd.read_csv('Data.csv')
#Creating matrix of dataset.
#X is an array of independent features
#y is an array of dependent results
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Taking care of missing values
#Imputer is a class which takes care of missing data.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy='mean', axis = 0)
#Fit this imputer data to matrix X.
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data(String -> Numbers)
#One object converts only one column of categorical data.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#OneHotEncoder is used for dummy encoding
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray() 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting dataset into training set and testing set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
#Must for accuracy and when our algo is dealing with Euclidean Distance.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 
