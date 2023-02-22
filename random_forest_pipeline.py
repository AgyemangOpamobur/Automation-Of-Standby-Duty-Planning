import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from preprocessing.preprocessor import DropFeatures, LogTransformer
from feature_engine.encoding import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import joblib
import os
import warnings
warnings.filterwarnings('ignore')
# import data

try:
    sickness = pd.read_csv("sickness_table.csv", parse_dates=['date'])
    # drop the first column
    sickness.drop('Unnamed: 0', axis=1, inplace=True)
    # preview the data
    sickness.head()
except IOError as err:
    print(err)

# create lag4 feature from the target
sickness['lag_4'] = sickness.sby_need.shift(4) # Shift the target 4 step
sickness = sickness.dropna() # Drop the rows with null values

# create rolling mean of 7 feature
sickness['rolling_mean'] = sickness.sby_need.rolling(7).mean()
sickness = sickness.dropna() # Drop the rows with null values
sickness.head() # preview the new addition

# Make the last 42 days a testset and the remaining days a trainset.
# make the date the index
sickness = sickness.set_index(sickness.date)
X = sickness.drop('sby_need', axis=1) # get the independent variables
y = sickness.sby_need # get the target variable
step = 42 # Number of days from April16 to May27
X_train, X_test = X.iloc[:-step, :],X.iloc[-step:, :]
y_train, y_test = y.iloc[:-step], y.iloc[-step:]

# check the starting and ending date and the length of the train and testsets.
print(f"Train dates: {X_train.index.min()} --- {X_train.index.max()} (n={len(X_train)})")
print(f"Test dates: {X_test.index.min()} --- {X_test.index.max()} (n={len(X_test)})")

# Plot the train and test targets
fig, ax = plt.subplots(figsize=(9, 4))
y_train.plot(ax=ax, label='train') # plot train data target
y_test.plot(ax=ax, label='test') # plot test data target
ax.legend()
plt.show()
# log transform the target
y_train = np.log(y_train + 1)
y_test = np.log(y_test + 1)

# extract date time features
# Create function to get German Pubic Holidays
def get_holidays():
    "'The function out puts all public holidays in the Federal Republic of Germany from 2016 to 2019."
    # Import the required libraries
    import holidays
    from workalendar.europe import Germany
    holiday_list = [] # list for all the dates of public holidays
    # Get list of public holidays in Germany
    for i, v in holidays.Germany(years=[2016, 2017, 2018, 2019]).items():
        holiday_list.append(i) # append only the date and not the name
    return holiday_list

def date_features(df, holiday_list):
    '''Perform date operation on pandas dataframe and extract date features'''
    df['year'] = df['date'].dt.year.astype(str) # Get year from the date
    df['month'] = df['date'].dt.month.astype(str) # get months from the year
    df['day'] = df['date'].dt.day.astype(str) # Get days of the month
    df['week'] = df['date'].dt.week.astype(str) # Get weeks of the year
    df['week_day'] = df['date'].dt.day_name() # Get the days of the week
    df['is_weekend'] = df['week_day'].apply(lambda x: 1 if x in ['Sunday', 'Saturday'] else 0).astype(str) # specify if day isweekend or not
    df['holidays'] = df['date'].apply(lambda x: 1 if x in holiday_list else 0).astype(str)
    df = df.set_index(df.date) # set the date column as index

    # drop the date column
    df.drop('date', axis=1, inplace=True) # drop the date column
    return df # Return a new dataframe with the newly engineered features.

# get holidays
holidays = get_holidays()
# get date features for Trainingset
X_train = date_features(X_train, holidays)
X_train.head(2)
# get date features for Testset
X_test = date_features(X_test, holidays)
X_test.head(2)
# save the X_train,y_train,X_test,y_test for feature selection
X_train.to_csv('xtrain.csv', index=False)
X_test.to_csv('xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False)
y_test.to_csv('ytest.csv', index=False)

# Configuration
# categorical variables to encode
CATEGORICAL_VARS = ['month', 'week_day',
'day', 'year', 'is_weekend', 'holidays']
# features to drop
REF_VARS = ['week', 'n_duty', 'n_sby', 'dafted']
# features to log transform
LOG_VARS = ['lag_4', 'rolling_mean']
# setup the pipeline
sby_need_pipe = Pipeline([

    # ====VARIABLE TRANSFORMATION=====
    ('log', LogTransformer(LOG_VARS)),
    # encode categorical and discrete variables using the target mean
    ('categorical_encoder', OneHotEncoder(variables=CATEGORICAL_VARS)),
    ('drop_features', DropFeatures(REF_VARS))
])

# train the pipeline
sby_need_pipe.fit(X_train, y_train)
# transform x_train
X_train = sby_need_pipe.transform(X_train)
# transform x_test
X_test = sby_need_pipe.transform(X_test)

# save pipeline
joblib.dump(sby_need_pipe, 'sby_need_pipe')
# regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
joblib.dump(rf_model,'random_forest_model')

# evaluate the model:
# ====================
# make predictions for trainset
print('Random ForestModel')
print('-'*20)
print('Random forest prediction for trainset')
print()
pred = rf_model.predict(X_train)
# determine mse,rmse and r2
print('train mse: {}'.format(int(mean_squared_error(np.exp(y_train) - 1, np.exp(pred) - 1))))
print('train rmse: {}'.format(int(mean_squared_error(np.exp(y_train) - 1, np.exp(pred) - 1, squared=False))))
print('train r2: {}'.format(r2_score(np.exp(y_train) - 1, np.exp(pred) - 1)))
print()
print()

# make predictions for testset
print('Random Forestpredictionfortestset')
print()
pred1 = rf_model.predict(X_test)

# determine mse,rmse and r2
print('test mse: {}'.format(int(mean_squared_error(np.exp(y_test) - 1, np.exp(pred1) - 1))))
print('test rmse: {}'.format(int(mean_squared_error(np.exp(y_test) - 1, np.exp(pred1) - 1, squared=False))))
print('test r2: {}'.format(r2_score(np.exp(y_test) - 1, np.exp(pred1) - 1)))

# let's evaluate our predictions respect to the sby_need
y = np.exp(y_test) - 1 # reverse the log transform of the target
y = y.reset_index() # reset the index
pred = rf_model.predict(X_test) # get predictions
pred = pd.Series(np.exp(pred) - 1) # reverse log transformation

# plot
# Plot the test and prediction targets
fig, ax = plt.subplots(figsize=(9, 4))
y['sby_need'].plot(ax=ax, label='Real') # plot test set
pred.plot(ax=ax, label='Predictions') # plot predicted target
ax.legend()

# Score new data
# load the unseen/new dataset
data = pd.read_csv('xtest.csv')
print(data.head())

# check shape
print('data shape:', data.shape)

# load feature engineering pipeline
sby_need_pipe = joblib.load('sby_need_pipe')

# load random forest model
rf_model = joblib.load('random_forest_model')

# transform the dataset with a feature engineering pipeline
data = sby_need_pipe.transform(data)

# check shape
print('Transform datashape:', data.shape)
# predict new score
prediction = rf_model.predict(data)

# print prediction
pred = pd.Series(np.exp(prediction) - 1)
print(pred)



