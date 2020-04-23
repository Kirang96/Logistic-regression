# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:40:12 2020

@author: Kiran
"""

import pandas as pd
import numpy as np

heartdatasample = pd.read_csv('heart.csv')
heartdata = pd.read_csv('heart.csv')
heartdata.shape
details = heartdata.describe()

#normalising
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
heartdata.iloc[:,[0,2,3,4,7,9]]=scaler.fit_transform(heartdata.iloc[:,[0,2,3,4,7,9]])
heartdata

'''
#standardising
from sklearn.preprocessing import StandardScaler
standarder = StandardScaler()
heartdata.iloc[:,[0,2,3,4,7,9]]=standarder.fit_transform(heartdata.iloc[:,[0,2,3,4,7,9]])
'''
x = heartdata.iloc[:,:-1]
y = heartdata.iloc[:,-1]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=101)


import matplotlib.pyplot as plt
plt.scatter(heartdata['age'],heartdata['cp'])


'''
from sklearn.linear_model import LinearRegression
algorithm = LinearRegression()
algorithm.fit(x_train, y_train)
new_values = algorithm.predict(x_test)
'''

from sklearn.tree import DecisionTreeRegressor
algorithm = DecisionTreeRegressor()
algorithm.fit(x_train, y_train)
new_values = algorithm.predict(x_test)


from sklearn.metrics import r2_score
result = r2_score(y_test,new_values)*100
result
