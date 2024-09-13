import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('data/creditcard.csv')

def data_info(data):
    #data.head(5)
    #print(data.describe())
    print(data.info)

def explain_data_frauds():
    """PRINT DATA DESCRIBING PERCENTAGE OF FRAUDS AND VALIDS"""
    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    #print(len(data[data['Class'] == 1]))
    #print(len(valid))

    portionfrauds = len(fraud)/len(valid)
    print(portionfrauds)

    print("Fraud cases:", len(fraud))
    print("Valid transactions:", len(valid))
    print("Portion of frauds", portionfrauds,"this means that ", portionfrauds*100, "are frauds")
    amount_describe(valid)

def amount_describe(valid):
    """PRINT INFORMATIONS ABOUT THE GIVEN PARAMETER"""
    print(valid.Amount.describe())

def correlation_matrix():
    """CORRELATION MATRIX AND HEATMAP"""
    fig = plt.figure(figsize=(20,20))
    sns.heatmap(data.corr(), vmax = 0.6, square=True)
    plt.show()

def valid_and_train():
    """BUILD MY DATA"""
    x = data.drop(['Class'],axis = 1)
    y = data['Class']

    print(x.shape)
    print(y.shape)
    
    xData = x.values
    yData = y.values
    train_model(xData, yData)

def train_model(xData, yData):
    xtrain, xtest, ytrain, ytest = train_test_split(xData, yData, test_size=0.2, random_state=42)
    rfc = RandomForestClassifier()
    #print("Valores de xtrain",xtrain)
    #print("Valores de ytrain",ytrain)

    rfc.fit(xtrain,ytrain)
    ypred = rfc.predict(xtest)
    print(ypred)


#data_info(data)
#explain_data_frauds()
correlation_matrix()
#valid_and_train()