#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:35:05 2019

@author: tapan
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Creating Dataframe for both csv file sets

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# Column count
print(train_df.columns)
train_df.isnull().sum()



######   Considering only few coloumns ############
###################################################

# pull data into target (y) and predictors (X)
train_y = train_df.SalePrice
#predictor_cols = ['LotArea', 'OverallQual','OverallCond', 'TotalBsmtSF','FullBath','GarageArea', 'TotRmsAbvGrd']
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train_df[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test_df[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test_df.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


##################################################
##################################################



