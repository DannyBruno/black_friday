"""

Initial Regressions and Visualizations. Expanded upon further in model.py, analysis.py.

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.style.use('seaborn')


def select_features(df):
    """
    Train/test split. Doesn't really need to be set out like this but this was a first attempt.
    """

    # 70/30 train/test.
    X_train, X_test = train_test_split(df[['Gender', 'Age', 'Occupation', 'City_Category', 'Marital_Status']], test_size=.3)
    y_train, y_test = train_test_split(df[['Purchase']], test_size=.3)

    return X_train, X_test, y_train, y_test


def plot_results(y_test, y_predicted):
    """
    Helper function for visualizing performance.
    """

    #print(f'Avg. distance between predictions and real values: {np.mean(np.abs(y_test - y_predicted))}')

    plt.plot(range(y_test.shape[0]), y_test, label="Real Purchase Amount")
    plt.plot(range(y_predicted.shape[0]), y_predicted, label="Predicted Purchase Amount")

    plt.xlabel("Customer")
    plt.ylabel("Purchase Amount")
    plt.title("Model Results")
    plt.legend(loc='best')
    #plt.savefig('plots/poly_regression.png')
    plt.show()



def poly_regression(df, degree):
    """
    Function to perform regression of various degrees. Really was not the right data set for it in hindsight.
    """

    X_train, X_test, y_train, y_test = select_features(df)

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    regress = LinearRegression().fit(X_train_poly, y_train)
    y_predicted = regress.predict(X_test_poly)

    print('Coefficients:')
    print(regress.coef_)

    plot_results(y_test, y_predicted)



if __name__ == '__main__':
    poly_regression(pd.read_csv("data/BlackFriday.csv", nrows=50000), 1)

