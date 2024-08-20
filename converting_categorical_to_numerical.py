import numpy as np
import pandas as pd
from Data import readfile
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce

df=readfile()

def frequency_encoder(df,column):
    frequency=df[column].value_counts()
    df[column]=df[column].map(frequency)
    return df

def converting_categorical():
    object_columns = df.select_dtypes(include='object')
    for i in object_columns:
        print(i,':',df[i].unique())
    print("The values which u see are the categorical values, and we need to convert them to numerical columns.")
    print("Ways to convert are:")
    print("1.Label Encoding ")
    print("2.Ordinal Encoding")
    print("3.Frequency Encoding")
    while True:
        y=input("Please type out the number of  which encoding method u prefer:")
        if y == "Label Encoding":
            label_encoder = LabelEncoder()
            for i in object_columns:
                df[i] = label_encoder.fit_transform(df[i])
            return df
        
        elif y == "Ordinal Encoding":
            ordinal_encoder=OrdinalEncoder()
            for i in object_columns:
                df[i]=ordinal_encoder.fit_transform(df[[i]])
            #print(df.info())
            return df
        elif y == "Frequency Encoding":
            for i in object_columns:
                df1=frequency_encoder(df,i)
            return df1
    
        else:
            print("Please check if the encoding method is in options or not! (Retype)")
        #print("this is the information for the frequency encoded dataset:",df1.head())
    
    #print("Done with converting categorical variables to numerical")



#converting_categorical()