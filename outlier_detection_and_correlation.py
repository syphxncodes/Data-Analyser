import numpy as np
import pandas as pd
import seaborn as sns
from filling_null_values import nullvalues
import matplotlib.pyplot as plt
import time

df=nullvalues()
def outliers():
    for col in df.columns:
        sns.boxplot(x=col,data=df)
        plt.show()


print("Null values have been removed:",df.isnull().sum())
print("Now, let us see if any outliers are there or not")
time.sleep(5)

outliers()
print("U will be getting the option to scale the data at a later stage, this is just for an analysis.")


print("This is the correlation matrix for the given dataset:")
matrix=df.corr()
sns.heatmap(matrix)
plt.show()



def graphs():
    selection=input("Do you want to view any statistics?(y/n)")
    if selection=="y":
        print('''Graphs available: 
          1)Scatterplot
          2)Line Plot
          3)Histogram
          4)KDE
          5)Bar Plot
          6)Pair Plot
          7)Joint Plot
          8)Reg Plot''')
        while True:
            selection_1=input("Please type in what u require:")
            if selection_1 == "Scatterplot":
                x=input("Enter x column:")
                y=input("Enter y column:")
                sns.scatterplot(x=x,y=y,data=df)
                plt.show()
                break
        
            elif selection_1 == "Line Plot":
                x=input("Enter x column:")
                y=input("Enter y column:")
                sns.lineplot(x=x,y=y,data=df)
                plt.show()
                break
        
            elif selection_1 == "Histogram":
                x=input("Enter x column:")
                y=input("Enter y column:")
                sns.histplot(x=x,y=y,data=df)
                plt.show()
                break

            elif selection_1 == "KDE":
                x=input("Enter x column:")
                y=input("Enter y column:")
                sns.kdeplot(x=x,y=y,data=df)
                plt.show()
                break
            
            elif selection_1 == "Bar Plot":
                x=input("Enter x column:")
                y=input("Enter y column:")
                sns.barplot(x=x,y=y,data=df)
                plt.show()
                break

            elif selection_1 == "Pair Plot":
                x=input("Enter x column:")
                y=input("Enter y column:")
                sns.pairplot(data=df)
                plt.show()
                break

            elif selection_1 == "Joint Plot":
                x=input("Enter x column:")
                y=input("Enter y column:")
                sns.jointplot(x=x,y=y,data=df)
                plt.show()
                break

            elif selection_1 == "Reg Plot":
                x=input("Enter x column:")
                y=input("Enter y column:")
                sns.regplot(x=x,y=y,data=df)
                plt.show()
                break
            else:
                print("No such option in the given list. Please recheck the spelling mistake (case sensitive)")
    else:
        pass
    return df

