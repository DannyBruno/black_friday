'''

Model framework.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics, RandomizedSearchCV
from scipy.stats import sp_randint

# Models.
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from regressions import plot_results

target = 'Purchase'
IDcol = ['User_ID', 'Product_ID']


'''
Cross-Validation utilities.
'''

def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(f"Mean validation score: {results['mean_test_score'][candidate]} (std: {results['std_test_score'][candidate]}")
            print(f"Parameters: {results['params'][candidate]}")



# Do I want to pick the best one from cross-validation or do I want to graph and see where overfitting starts to occur? Should have same effect.
# Decision tree CV. Max depth, min samples split, min samples leaf,
def DTR_CrossVal(alg, X, y):

    # Parameters to select from.
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 20),
                  "bootstrap": [True, False],
                  "criterion": ['gini', 'entropy']
                  }

    alg = RandomizedSearchCV(alg, param_dist=param_dist, n_iter=20, cv=5, iid=False)
    report(alg.cv_results_)

    alg.fit(X, y)


'''
Modelling.
'''

def modelfit(alg, data, features, target):

    # Split data into train and test.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data[features], data[target],
                                       test_size=.2)

    # Baseline fit.
    alg.fit(X_train, y_train)
    train_predictions = alg.predict(X_train)
    print("RMSE Train: %.4g" % np.sqrt(metrics.mean_squared_error(y_train, train_predictions)))


    # Cross_validation/hyper-parameter selection.
    DTR_CrossVal(alg, X_train, y_train)

    # Report results + visualize.
    test_predictions = alg.predict(X_test)
    print("RMSE Test: %.4g" % np.sqrt(metrics.mean_squared_error(y_test, test_predictions)))

    plot_results(y_test, test_predictions)





if __name__ == '__main__':

    # Read in data.
    df = pd.read_csv("data/BlackFriday_Modified.csv", nrows=50000)
    features = df.columns.drop(['Purchase', 'Product_ID', 'User_ID'])

    """
    # Linear Regression Model. (Similar to regressions.py)
    print("Linear Regression Model")
    LR = LinearRegression(normalize=True)
    modelfit(LR, df, features, target)

    # What is contributing?
    coef1 = pd.Series(LR.coef_, features).sort_values()
    #coef1.plot(kind='bar', title='LR Coeffs')
    #plt.show()

    # Ridge Regression Model.
    print("Ridge Regression Model")
    RR = Ridge(alpha=.05, normalize=True)
    modelfit(RR, df, features, target)

    coef2 = pd.Series(RR.coef_, features).sort_values()
    coef2.plot(kind='bar', title='RR Coeffs')
    plt.show()

    # Lasso.
    print("Lasso Model")
    LM = Lasso(alpha=.01, normalize=True)
    modelfit(LM, df, features, target)

    coef3 = pd.Series(LM.coef_, features).sort_values()
    coef3.plot(kind='bar', title='LM Coeffs')
    plt.show()
    """
    # Poly. regression found in other file. Performs similarly poorly.

    # Decision Tree Regressor. (Now testing more complex models.) (These all need some hyperparameter tuning.)
    print("Decision Tree Regressor")
    DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
    modelfit(DT, df, features, target)

    """
    # Random Forest Regressor.
    print("Random Forrest Regressor")
    RF = RandomForestRegressor(n_estimators=20, max_depth=15)
    modelfit(RF, df, features, target)
    """

    """
    TODO: 
        - Add a network. 
        - Do more sophisticated feature selection (Network and KNN feature imputation + more hotencoding for categorical).
        - Actually use the cross-validation for hyper-parameter tuning.
    """

    """
     # For each one assess using cross val score.
    cv_score = model_selection.cross_val_score(alg, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    print("CV Score: Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
    """







