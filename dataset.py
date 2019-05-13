"""

Initial dataset preprocessing.

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# Tables.
AGE_TABLE = {
    "0-17": 0,
    "18-25": 1,
    "26-35": 2,
    "36-45": 3,
    "46-50": 4,
    "51-55": 5,
    "55+": 6,
    np.NAN: -1
}

GENDER_TABLE = {
    'F': 0,
    'M': 1,
    np.NAN: -1
}

CITY_TABLE = {
    'A': 1,
    'B': 2,
    'C': 3,
    np.NAN: -1
}


def clean_data(row_data):
    """
    Clean up data.

    """
    # Gender.
    gender = GENDER_TABLE[row_data['Gender']]

    # Age.
    age = AGE_TABLE[row_data['Age']]

    # City category.
    city_category = CITY_TABLE[row_data['City_Category']]


    # City stay.
    city_stay = row_data['Stay_In_Current_City_Years']
    if city_stay == '4+':
        city_stay = 4

    return [gender, age, city_category, city_stay]



def impute_missing_vals(method, row_data):
    """
    Data imputation methods.
    """
    # Try several methods for imputation.

    # Method 0: Replace Nans w/ -2s.
    if method == 0:
        row_data.fillna(-1)

    # Method 1: Predict using Network.
    elif method == 1:
        assert 0

    # Method 2: Predict using KNN.
    else:
        assert 0

    return row_data[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']]



def export(df):
    df.to_csv('data/BlackFriday_Modified.csv', index=False)


def load_data():
    # Import data.
    df = pd.read_csv("data/BlackFriday.csv") #nrows=10000

    # Confirm columns with null values.
    print(df.isnull().sum())

    # Fill in missing values with -2s.
    df[['Product_Category_2']] = df[['Product_Category_2']].fillna(-2.0).astype(float)
    df[['Product_Category_3']] = df[['Product_Category_3']].fillna(-2.0).astype(float)


    df[['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']] = df.iloc[:, :].progress_apply(lambda x: pd.Series(clean_data(x)), axis=1)
    print("Gender, Age, and Current Stay Corrected!")

    # Remove categories that have few entries.
    # Maybe come back to this.
    #df[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']] = df.iloc[:, :].progress_apply(lambda x: pd.Series(impute_missing_vals(0, x)), axis=1)
    #print("Missing Values Imputed!")


    print("Exporting!")
    export(df)


if __name__ == '__main__':
    load_data()
