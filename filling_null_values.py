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
    print("4. Interpolation")
    print("5. Remove the null values")
    while True:
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
        elif selected_option=="Interpolation":
            for col in df.columns:
                if col==x:
                    continue
                else:
                    df.interpolate(method="linear",inplace=True)
            return df
        elif selected_option == "Remove the null values":
            df.dropna(axis=1,inplace=True)
            return df
        else:
            print("Please retype by looking at the given option (Recheck spelling)")

    #print("Done with filling null values with the required selection")

#df1=nullvalues()
#print(df1.isnull().sum())