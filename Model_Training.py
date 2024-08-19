import numpy as np
import pandas as pd
from filling_null_values import nullvalues
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
#import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

df=nullvalues()
selection=input("Based on the statistics that you viewed, do u want to remove any column(y/n):")

if selection == 'y':
    selection_2=int(input("Please input the number of columns to remove:"))
    for i in range(0,selection_2):
        column=input("Please input the column name:")
        df.drop(columns=column,inplace=True)
        print("The column is removed, here is the dataframe information:")
        print(df.info())
else:
    pass


print("Now we will come to train test split")

def ModelTraining():
    input1=input("Enter the column that needs to be trained:").replace("'","").replace('"',"")
    x=df.drop(columns=[input1])
    y=df[input1]
    random_state_input=int(input("Enter the random state required:"))
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=random_state_input)
    print("Train Test Split is done")
    print(y.unique())
    input1=input("Please type what problem it is based on(Classification or Regression):")
    if input1 == "Classification":
        print("The default metric for classification used is accuracy")
        models=[LogisticRegression(),KNeighborsClassifier(),DecisionTreeClassifier(),GradientBoostingClassifier(),RandomForestClassifier(),GaussianNB(),MLPClassifier(),AdaBoostClassifier()]
        #models=KNeighborsClassifier()
        accuracies={}
        for i in models:
            model=i
            model.fit(xtrain,ytrain)
            y_pred=model.predict(xtest)
            accuracy=accuracy_score(ytest,y_pred)
            accuracies[model]=accuracy
        print(accuracies)

ModelTraining()