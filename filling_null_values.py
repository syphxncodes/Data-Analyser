from main import readfile
import numpy as np
import pandas as pd
from converting_categorical_to_numerical import converting_categorical

df=converting_categorical() #gives u the database
x=input("Enter the name of the column to be trained:")

#print(df.isnull().sum())
def nullvalues():
    print("These are the columns that have null values:")
    print(df.isnull().sum())
    print("These are the available options to fill the null values if any:")
    print("1. Mean")
    print("2. Mode")
    print("3. Median")
    selected_option=input("Type your required way to fill the null values:")
    if selected_option=="Mean":
        for col in df.columns:
            if col==x:
                continue
            else:
                df[col]=df[col].fillna(df[col].mean())
        return df
    
    elif selected_option=="Median":
        for col in df.columns:
            if col==x:
                continue
            else:
                df[col]=df[col].fillna(df[col].median())
        return df
    
    elif selected_option=="Mode":
        for col in df.columns:
            if col==x:
                continue
            else:
                df2= df.apply(lambda col: col.fillna(col.mode()[0]))
            
        return df2

df1=nullvalues()
print(df1.isnull().sum())