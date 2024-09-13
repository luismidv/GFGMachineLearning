import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("data/WineQT.csv")

def get_info():
    """DIFFERENT LINES TO GET INFORMATION ABOUT THE DATA
    """
    #data.info()
    #print(data.describe().T)
    print(data.isnull().sum())

def check_null_values():
    """1.THIS FUNCTION SEARCH FOR NULL DATA IN OUR CSV. 
       2.IF FOUND THE NULL VALUE IS REPLACED WITH THE COLUMN'S MEAN, SO WE DON'T MAKE A BIG DIFFERENCE ON THE RESULT.
    """
    i = 0
    for col in data.columns:
        i+=1
        if data[col].isnull().sum() > 0:
            print("Column with null value found", " Col Nº", i, " Total Columns", len(data.columns))
            data[col] = data[col].fillna(data[col].mean())
        else:
            print("This column has no null value ", "Col Nº", i, " Total Columns", len(data.columns))

def data_visualization():
    """VISUALIZATION OF DIFFERENT DATA USING MATPLOTLIB
    """
    #data.hist(bins = 20, figsize=(10,10))
    plt.figure(figsize=(14,9))
    plt.bar(data['quality'], data['alcohol'], width = 0.5, align= 'edge')
    plt.xlabel("Quality")
    plt.ylabel("Alcohol")
    #plt.show()

def redundant_data():
    """SOMETIMES THE DATA GIVEN CONTAINS REDUNDANT CHARACTERISTICS THAT DOESN'T HELP MODEL EFFICIENCY
       NEXT STEP IS TO FIND AND REMOVE IT 
       IN THIS CASE THE CORRELATION VALUE NEEDS TO BE SET TO 0.6 TO FIND A CORRELATION
       IT WOULD BE A MOST CONSIDERABLE CORRELATION IF WE FOUND IT WITH 0.7 VALUE, LET'S PASS IT FOR THIS TIME"""
    
    plt.figure(figsize=(12,12))
    sns.heatmap(data.corr() > 0.6, annot = True, cbar = False)
    plt.show()
    
    new_data = drop_columns("total sulfur dioxide", data)
    new_data = drop_columns("fixed acidity", new_data)
    return new_data

def drop_columns(column,data):
    """DROP COLUMNS THAT ARE REDUNDANT"""
    new_data = data.drop(column, axis = 1)
    #print("Data after delete\n",new_data)
    return new_data

def add_best_quality(data):
    """ADDING A COLUMN TO CONTROL WINES WITH BEST QUALITY"""
    data['best quality'] = [1 if x > 5 else 0 for x in data.quality]
    
    #print(data.head(7))
    return data
"""
def replace_data_obj(data):
    print("Trying to replace data")
    try:
        data.replace({'white': 1, 'red' : 0}, inplace = True)
    except:
        print("Nothing replaced")
    print(data)
"""

def build_model_data(data):
    """BUILD NECESSARY DATA FOR ML MODEL
       XTRAIN ARE DATA USED CHARACTERISTICS USED TO TRAIN THE MODEL
       XTEST IS THE VALIDATION DATA"""
    features = data.drop(['quality', 'best quality'], axis = 1)
    target = data['best quality']
    xtrain, xtest, ytrain, ytest = train_test_split(features,target,test_size = 0.2, random_state = 40)
    print("Set de train y test")
    #print(xtrain)
    #print(xtest)
    
    normalize(xtrain,xtest,ytrain,ytest)

def normalize(xtrain,xtest,ytrain,ytest):
    """FIT_TRANSFORM IS APPLIED TO TRAINING DATA
       TRANSFORM IS APPLIED TO VALIDATION DATA
       WE CAN USE A DIFFERENT SCALER, IT DEPENDS ON WHAT WE NEED
       IN THIS CASE WE CAN CHOOSE A RANGE OF VALUES OR USE STANDARD SCALER
       FOR VALUES TO BE AROUND 0"""
       
    norm = MinMaxScaler()
    print(xtrain)
    xtrain = norm.fit_transform(xtrain)
    print(xtrain)
    xtest = norm.transform(xtest)

    train_model(xtrain,xtest,ytrain,ytest)

def train_model(xtrain,xtest,ytrain,ytest):
    print("Xtrain\n", xtrain)
    print("Xtest\n", xtest)
    print("ytrain\n", ytrain)
    print("ytest\n", ytest)
    models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
    """
    for i in range(3):
        
        models[i].fit(xtrain,ytrain)
        print(f'{models[i]}')
        print("Training accuracy:", metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
        print("Validation accuracy:", metrics.roc_auc_score(
        ytest, models[i].predict(xtest)))
    """

#get_info()
#col_check()
#check_null_values()
#data_visualization()
data = redundant_data()
data = add_best_quality(data)
#data = replace_data_obj(data)
build_model_data(data)

