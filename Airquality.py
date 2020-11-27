#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:29:19 2020

@author: aditisaxena
"""
#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import train and test set
dataset=pd.read_csv('Train.csv')
testdata=pd.read_csv('Test.csv')

X_train=dataset.iloc[:,:-1].values
y_train=dataset.iloc[:,5].values

#Train using Multiple regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predict output for test set
y_pred=regressor.predict(testdata)

