#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:35:05 2019

@author: tapan
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Creating Dataframe for both csv file sets

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# Column count
print(train_df.columns)
train_df.isnull().sum()



# pull data into target (y) and predictors (X)
train_y = train_df.SalePrice
#predictor_cols = ['LotArea', 'OverallQual','OverallCond', 'TotalBsmtSF','FullBath','GarageArea', 'TotRmsAbvGrd']
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']


# Create training predictors data
train_X = train_df[predictor_cols]

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)




my_model_linear = LinearRegression()
my_model_linear.fit(X_train, y_train)


my_model_randomforest_reggressior = LinearRegression()
my_model_randomforest_reggressior.fit(X_train, y_train)


## Treat the test data in the same way as training data. In this case, pull same columns.
#test_X = test_df[predictor_cols]
## Use the model to make predictions
#predicted_prices = my_model.predict(test_X)
## We will look at the predicted prices to ensure we have something sensible.
#print(predicted_prices)


predicted_prices_linear = my_model_linear.predict(X_test)

print(predicted_prices_linear)


predicted_prices_randomforest_reggressior = my_model_randomforest_reggressior.predict(X_test)

print(predicted_prices_randomforest_reggressior)

#print("Accuracy:",metrics.accuracy_score(y_test, predicted_prices))
#score  = accuracy_score(y_test, predicted_prices)



my_submission = pd.DataFrame({'Id': X_test.index, 'SalePrice': predicted_prices_linear})
 #you could use any filename. We choose submission here
my_submission.to_csv('linear_submission.csv', index=False)



#




