import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from fileOperations import FileOperations
from configReader import ConfigReader
from src.preProcess import Preprocess
from src.featureEngineer import FeatureEngineer
from src.splitData import split_data

from tqdm import tqdm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import ensemble

fileOperator = FileOperations()
configOperator = ConfigReader()

rawDataFrame = fileOperator.load_raw_dataset(configOperator.inputDirectory + configOperator.inputFile)

processedDataFrame = \
    Preprocess(rawDataFrame) \
    .get_null_values_per_column() \
    .remove_null_values('target') \
    .fill_null_values(['far_price', 'near_price'], 0.0) \
    .fill_null_values(type_fill="mean") \
    .df

processedDataFrame.loc[processedDataFrame['imbalance_buy_sell_flag'] == 0.0, 'imbalance_size'] = 0.0

engineeredDataFrame = \
    FeatureEngineer(processedDataFrame)\
    .generate_spread() \
    .generate_imbalance_ratio() \
    .generate_volume()\
    .generate_mid_price()\
    .liquidity_imbalance()\
    .match_ratio()\
    .generate_minutes()\
    .df

print("\nRaw DataFrame Example Data")
print(rawDataFrame.head())

print("\nNull Values in Raw DataFrame")
Preprocess(rawDataFrame).get_null_values_per_column()

print("\nData Frame After Feature Engineering")
print(engineeredDataFrame.head())

print("\nNow splitting the dataset into Training and Test Sets based on a 80/20 split.")
trainX, trainY, crossValX, crossValY = split_data(engineeredDataFrame)

print("Training the following Machine Learning Models:")
print("1. Linear Regression")
linearRegModel = linear_model.Ridge(alpha=.5)
linearRegModel.fit(trainX, trainY)
linearTarget = linearRegModel.predict(crossValX)
lTError = mean_squared_error(crossValY, linearTarget)

# print("2. Elastic Net")
# elasticNetModel = linear_model.ElasticNet(alpha=.5, tol=0.001)
# elasticNetModel.fit(trainX, trainY)
# elasticNetTarget = elasticNetModel.predict(crossValX)
# eNTError = mean_squared_error(crossValY, elasticNetTarget)

print("3. K Nearest Neighbors")
knnModel = neighbors.KNeighborsRegressor(n_neighbors=5, weights="uniform", n_jobs=5)
knnModel.fit(trainX, trainY)
knnModelTarget = knnModel.predict(np.array(crossValX.iloc[1]).reshape(1, -1))
kMTError = mean_squared_error(np.array(crossValY.iloc[1]).reshape(1, -1), knnModelTarget)

print("4. Random Forest")
randomForestModel = ensemble.RandomForestRegressor(24, max_samples=0.7, n_jobs=5)
randomForestModel.fit(trainX, trainY)
randomForestTarget = randomForestModel.predict(crossValX)
rFTError = mean_squared_error(crossValY, randomForestTarget)

print("Here are the generated statistics for the tested algorithms on the cross validation set.")
print("MSE for Linear Regression:", lTError)
print("MSE for K Nearest Neighbors:", kMTError)
print("MSE for Random Forest:", rFTError)

# fileOperator.save_prep_dataset(engineeredDataFrame, configOperator.outputDirectory + configOperator.outputFile)
