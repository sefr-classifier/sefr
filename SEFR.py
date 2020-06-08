import numpy as np
import pandas as pd
import statistics

weights = []
bias = 0

# This is used for training the classifier on data. train_predictors are the main data, and train_target are the labels. 
# The labels should be 0 or 1.
def fit(train_predictors, train_target):

    neglabels = train_target == 0
    # poslabels are those records where the label is positive
    # neglabels are those records where the label is negative

    poslabels = train_target == 1
    neglabels = train_target == 0

    # dfpos are the data where the labels are positive
    # dfneg are the data where the labels are negative
    dfpos = train_predictors.loc[poslabels] 
    dfneg = train_predictors.loc[neglabels] 

    # avgpos is the average value of each feature where the label is positive
    # avgneg is the average value of each feature where the label is negative
    avgpos = dfpos.mean(skipna=True) # Eq. 3
    avgneg = dfneg.mean(skipna=True) # Eq. 4

    # weights are calculated based on Eq. 3 and Eq. 4
    global weights
    weights = (avgpos - avgneg) / (avgpos + avgneg) #Eq. 5
    weights.fillna(0, inplace=True)

    posscore = []
    negscore = []

        
    # For each record, a score is calculated. If the record is positive/negative, the score will be added to posscore/negscore
    for i in range(len(train_predictors.index)):
        temp = np.dot(weights, train_predictors.iloc[i,:]) #Eq. 6
        if train_target.iloc[i] == 0: 
            negscore.append(temp)
        if train_target.iloc[i] == 1: 
            posscore.append(temp)

    # posscoreavg and negscoreavg are average values of records scores for positive and negative classes
    posscoreavg = statistics.mean(posscore) # Eq. 7
    negscoreavg = statistics.mean(negscore) # Eq. 8

    # bias is calculated using a weighted average
    global bias

    bias = (len(negscore) * posscoreavg + len(posscore) * negscoreavg) / (len(negscore) + len(posscore)) # Eq. 9

    
# This is for prediction. When the model is learned, it cal be applied on the test data. test_predictors are the data without labels.
def predict(test_predictors):

    pred = []
        if temp <= bias: pred.append(0)
        if temp > bias: pred.append(1)

        # The list of predicted labels are returned
    return pred
