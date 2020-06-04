import numpy as np
import pandas as pd
import statistics


weights = []
bias = 0
def fit(train_predictors, train_target):
    
    neglabels = train_target == 0
    poslabels = train_target == 1
            
    dfpos = train_predictors.loc[poslabels] 
    dfneg = train_predictors.loc[neglabels] 

    avgpos = dfpos.mean(skipna=True) # Eq. 3
    avgneg = dfneg.mean(skipna=True) # Eq. 4
    
    global weights
    weights = (avgpos - avgneg) / (avgpos + avgneg) #Eq. 5
    weights.fillna(0, inplace=True)

    posscore = []
    negscore = []
        
        
    for i in range(len(train_predictors.index)):
        temp = np.dot(weights, train_predictors.iloc[i,:]) #Eq. 6
        if train_target.iloc[i] == 0: 
            negscore.append(temp)
        if train_target.iloc[i] == 1: 
            posscore.append(temp)
    
    posscoreavg = statistics.mean(posscore) # Eq. 7
    negscoreavg = statistics.mean(negscore) # Eq. 8
    
    global bias

    bias = (len(negscore) * posscoreavg + len(posscore) * negscoreavg) / (len(negscore) + len(posscore)) # Eq. 9

        
    
def predict(test_predictors):
        
    pred = []
    temp = 0

    for i in range(len(test_predictors)):
       
        temp = np.dot(weights, (test_predictors.iloc[i]))

     
        if temp <= bias: pred.append(0)
        if temp > bias: pred.append(1)
        
    return pred



    

    
    