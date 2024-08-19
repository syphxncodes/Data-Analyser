# Welcome to the Data Analyser Project!

This project aims to ease data preprocessing and model training by automation. The user has to give the path to use the dataset for cleaning, analysis and training.

## Converting Categorical Variables to Numerical Variables
First, information of the dataset is shown to the user, and then object columns are displayed with unique values of each of the column. Then, different ways are given to the user for the conversion of categorical variables to numerical variables. Options available:
* Label Encoding: Assigns a categorical variable to a specific integer.

* Ordinal Encoding: Makes a ranking among the categorical variables and then assigns a value.

* Frequency Encoding: Finds the frequency of a particular categorical variable and assigns the value to it.

## Filling Null Values
Filling of null values is very important because some models cannot process without this crucial step. Columns with null values are shown to the user. Methods available:
* Mean: Finds the mean of the particular column and fills the null value with found mean value.

* Median: Finds the median of the particular column and fills the null value with found mean value.

* Mode: Finds the mode of the particular column and fills the null value with found mean value.

* Interpolation: Interpolates the column to find the null value.

* Remove the null values: Removes the null values, but not the best option because of data loss.

## Outlier Detection and Correlation
Understanding the columns with outliers gives us the information of how the data is spread, and what scaling can be done so that the model can give the best result. This is achieved using box plots. All the data points that are outside of the box plot are considered as outliers, and they are treated using different scaling methods which will be mentioned later. Also, a correlation matrix is generated for the dataset.

## Scaling and Model Training
Scaling is performed to bring all features to a similar scale, ensuring that no feature disproportionately influences the model due to its range. This helps prevent issues like overfitting to certain features and improves model performance, especially for distance-based algorithms.

Methods available for scaling:

* Standard Scaler: Normalizes data to a mean of 0 and standard deviation of 1.

* MinMax Scaler: Scales data to a range between 0 and 1 by adjusting the minimum and maximum values of each feature.

* MaxAbs Scaler: Scales the data by dividing each feature by its maximum absolute value.

* Robust Scaler: Scales features using statistics that are robust to outliers.

* Normalizer: Scales each datapoint so that the magnitude of each row becomes 1.

There are many models available for training, some of them being LogisticRegression, DecisionTree and so on.

## Using the code

First, prerequisites to download this particular application:
- [Git](https://git-scm.com/downloads) installed on your machine.
- [Python](https://www.python.org/downloads/) installed on your machine.

  ```bash
  git clone https://github.com/syphxncodes/Alzheimer-Disease-Detection.git
Change the directory of your command prompt to the cloned repo
  ```bash
    cd Data-Analyser
```
Create a virtual enviroment for this to run.
```bash
python -m venv venv
```
* On Windows:
 ```bash
  venv\Scripts\activate
```
* On macOS/Linux:
 ```bash
  source venv/bin/activate
```
Install Dependencies

```bash
pip install -r requirements.txt
```
Then run the Model_Training.py.

```bash
python Model_Training.py
```