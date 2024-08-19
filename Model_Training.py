import numpy as np
import pandas as pd
from filling_null_values import nullvalues
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
#import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer



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
    input2=input("Do you want scaling?(y/n)")
    if input2 == 'y':
        print('''These are the types of scaling available:
                 1)StandardScaler
                 2)MinMaxScaler
                 3)MaxAbsScaler
                 4)RobustScaler
                 5)Normalizer'''
              )
        scaling_method=input("Please type the scaling method required:")
        if scaling_method == "StandardScaler":
            standardscaler=StandardScaler()
            xtrain=standardscaler.fit_transform(xtrain)
            xtest=standardscaler.transform(xtest)
        elif scaling_method == "MinMaxScaler":
            minmax=MinMaxScaler()
            xtrain=minmax.fit_transform(xtrain)
            xtest=minmax.transform(xtest)
        elif scaling_method == "MaxAbsScaler":
            maxabs=MaxAbsScaler()
            xtrain=maxabs.fit_transform(xtrain)
            xtest=maxabs.transform(xtest)
        elif scaling_method == "RobustScaler":
            robust=RobustScaler()
            xtrain=robust.fit_transform(xtrain)
            xtest=robust.transform(xtest)
        elif scaling_method == "Normalizer":
            normalizer=Normalizer()
            xtrain=normalizer.fit_transform(xtrain)
            xtest=normalizer.transform(xtest)
        else:
            print("This is not in the options, please check if any spelling mistake!")
    else:
        pass
    
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
        best_model_name = max(accuracies, key=accuracies.get)
        best_model_accuracy = accuracies[best_model_name]
        print("This is the best model:",best_model_name)
        print("The best model's accuracy:",best_model_accuracy)

        #input2=input("Do you want to do a gridsearchCV for better options of model accuracy?(y/n)")
        #if input2 == 'y':
            #print(models)
            #model1=input("Type the model's name that u want to do gridsearchcv on:")
            #if model1 == "LogisticRegression":
                #param_grid = {
                    #'C': [0.1, 1, 10, 100],
                    #'solver': ['newton-cg', 'lbfgs', 'liblinear']
                #}
                #logreg=LogisticRegression(max_iter=100,random_state=42)
                #grid_search = GridSearchCV(logreg, param_grid, cv=5, n_jobs=-1, verbose=1)
                #grid_search.fit(xtrain,ytrain)

                #best_logreg=grid_search.best_estimator_
                #print(f"Best Parameters:{grid_search.best_params_}")
                #ypred=best_logreg.predict(xtest)
                #print("This is the best accuracy:",accuracy_score(ytest,ypred))

            #elif model1 == "KNeighborsClassifier":
                #param_grid = {
                    #'n_neighbors': [3, 5, 7, 9],          # Number of neighbors
                    #'weights': ['uniform', 'distance'],    # Weight function used in prediction
                    #'metric': ['euclidean', 'manhattan']   # Distance metric
                #}
                #knn=KNeighborsClassifier()
                #grid_search=GridSearchCV(knn,param_grid, cv=5, n_jobs=-1, verbose=1)
                #grid_search.fit(xtrain,ytrain)
                #best_knn = grid_search.best_estimator_
                #print(f"Best parameters: {grid_search.best_params_}")

                    # Evaluate on the test set
                #y_pred1 = best_knn.predict(xtest)
                #print(accuracy_score(ytest,y_pred1))
            

    elif input1 == "Regression":
        print("The defualt metric for regression used is f1-score")
        models1=[LinearRegression(),KNeighborsRegressor(),SVR(),DecisionTreeRegressor(),GradientBoostingRegressor(),RandomForestRegressor(),AdaBoostRegressor()]
        accuracies={}
        for i in models1:
            model=i
            model.fit(xtrain,ytrain)
            y_pred=model.predict(xtest)
            accuracy=accuracy_score(ytest,y_pred)
            accuracies[model]=accuracy
        print(accuracies)
        best_model_name = max(accuracies, key=accuracies.get)
        best_model_accuracy = accuracies[best_model_name]
        print("This is the best model:",best_model_name)
        print("The best model's accuracy:",best_model_accuracy)
    
ModelTraining()