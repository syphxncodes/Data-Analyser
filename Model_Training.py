import numpy as np
import pandas as pd
from filling_null_values import nullvalues
from sklearn.model_selection import train_test_split

df=nullvalues()
selection=input("Based on the statistics that you viewed, do u want to remove any column(y/n):")

if selection == 'y':
    selection_2=int(input("Please input the number of columns to remove:"))
    for i in range(0,selection_2):
        column=input("Please input the column name:")
        df.drop(columns=column,inplace=True)
        print("The column is removed, here is the dataframe information:")
        print(df.info())


print("Now we will come to train test split")

def ModelTraining():
    pass