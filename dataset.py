"""

Initial dataset preprocessing.

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from sklearn.tree import DecisionTreeClassifier

from interpolation import interpolation_model_fit, interpolation_tree, interpolation_mode_setup, interpolation_mode


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
        row_data.fillna(-2)

    # Method 1: Predict using Network.
    elif method == 1:
        assert 0

    # Method 2: Predict using KNN.
    else:
        assert 0

    return row_data[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']]



def export(df):
    #df.to_csv('data/BlackFriday_Modified.csv', index=False)
    #df.to_csv('data/BlackFriday_Modified_Interp.csv', index=False)
    df.to_csv('data/BlackFriday_Modified_Interp_Mode.csv', index=False)


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

    '''
    # Attempt to interpolate with tree.

    tree_two = DecisionTreeClassifier(max_depth=15, min_samples_leaf=50)
    tree_three = DecisionTreeClassifier(max_depth=15, min_samples_leaf=50)

    interpolation_model_fit(df, tree_two, tree_three)

    df[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']] = df.iloc[:, :].progress_apply(
        lambda row: pd.Series(interpolation_tree(row, tree_two, tree_three)), axis=1)
    '''
    # Attempt to interpolate with mode.

    modes_two = {}
    modes_three = {}

    modes_two, modes_three = interpolation_mode_setup(df, modes_two, modes_three)

    df[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']] = df.iloc[:, :].progress_apply(
        lambda row: pd.Series(interpolation_mode(row, modes_two, modes_three)), axis=1)

    print("Missing Values Imputed!")


    print("Exporting!")
    export(df)


if __name__ == '__main__':
    load_data()
