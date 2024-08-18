import numpy as np
import pandas as pd
import os
def readfile():
    path_of_file=input("Please provide the path of the file:")
    df=pd.read_csv(path_of_file)
    print("Information of the given csv file:")
    print(df.info())
    return df
