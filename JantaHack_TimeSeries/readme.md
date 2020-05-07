Problem Statement:
Train shape:(26496, 7),Test Shape:(8568, 6)
Predictor Variables:['datetime', 'temperature', 'var1', 'pressure', 'windspeed', 'var2']
Target Variable:['electricity_consumption']

Train data provided was first 23 days of every month from July 2013 to June 2017
Test data compromised of remaining 7 days of every month of the given timeperiod in train set.

Approach:
Based on EDA it was found there was seasonality in weekly,monthly and daily basis .So new features were created to cater to the same.
Featured added:['dayofweek', 'month', 'year',
       'is_month_end', 'is_month_start']
Log transformation of target variable was done as it was found to be skewed.
Using rolling window features were created and LGB algo applied to the data .
91.9964287743 RMSE score was obtained on LB with 25th Rank overall and 88.6 in public LB.
