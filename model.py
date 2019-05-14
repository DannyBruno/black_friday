'''

Model framework.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics
from scipy.stats import randint as sp_randint

# Models.
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from keras.models import Model
from keras.utils import plot_model
from keras.layers import (
    Input,
    Dense
)

from regressions import plot_results

target = 'Purchase'
IDcol = ['User_ID', 'Product_ID']


'''
Network Model.
'''

def build_network(weights_path: str=''):

    input_tensor = Input(shape=(9,))
    one = Dense(32)(input_tensor)
    two = Dense(64)(one)
    three = Dense(128)(two)
    four = Dense(128)(three)
    five = Dense(128)(four)
    six = Dense(64)(five)
    seven = Dense(32)(six)
    eight = Dense(8)(seven)
    output = Dense(1)(eight)


    model = Model(inputs=input_tensor, outputs=output)
    model.compile('adadelta', loss='mean_squared_error')

    if weights_path:
        model.load_weights(weights_path)

    return model


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


def dtr_cv(alg, X, y):

    # Parameters to select from. # These don't work well.
    param_dist = {"max_depth": [3, 50],
                  "max_features": sp_randint(1, 9),
                  "min_samples_split": sp_randint(50, 400),
                  "min_samples_leaf": sp_randint(50, 400),
                  "criterion": ['mse', 'friedman_mse', 'mae']
                  }

    alg = model_selection.RandomizedSearchCV(alg, param_distributions=param_dist, n_iter=3, scoring='neg_mean_squared_error', cv=5, iid=False)
    alg.fit(X, y)
    report(alg.cv_results_)

    train_predictions = alg.predict(X)
    print("RMSE Train (Post-CV): %.4g" % np.sqrt(metrics.mean_squared_error(y, train_predictions)))


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
    print("RMSE Train (Pre-CV): %.4g" % np.sqrt(metrics.mean_squared_error(y_train, train_predictions)))


    # Cross_validation/hyper-parameter selection.
    #dtr_cv(alg, X_train, y_train)

    # Report results + visualize.
    test_predictions = alg.predict(X_test)
    print("RMSE Test: %.4g" % np.sqrt(metrics.mean_squared_error(y_test, test_predictions)))

    plot_results(y_test, test_predictions)





if __name__ == '__main__':

    # Read in data.
    df = pd.read_csv("data/BlackFriday_Modified_Interp.csv")
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
    
    # Poly. regression found in other file. Performs similarly poorly.
    """

    # Decision Tree Regressor. (Now testing more complex models.) (These all need some hyperparameter tuning.)
    print("Decision Tree Regressor")
    DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
    modelfit(DT, df, features, target)

    """
    # Random Forest Regressor.
    print("Random Forrest Regressor")
    RF = RandomForestRegressor(n_estimators=20, max_depth=15)
    modelfit(RF, df, features, target)

    
    # Simple Network. # No one-hot encoding so this was dumb to try. Also explains poor performance in regressions.
    network = build_network()
    print("Simple Network")
    modelfit(network, df, features, target)
    #plot_model(network, to_file='plots/model.png')
    """

    """
    TODO: 
        /- Add a network. 
        - Do more sophisticated feature selection (Network and KNN feature imputation + more hotencoding for categorical).
        /- Actually use the cross-validation for hyper-parameter tuning. 
    """

    """
    Misc.
     # For each one assess using cross val score.
    cv_score = model_selection.cross_val_score(alg, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    print("CV Score: Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
    # Do I want to pick the best one from cross-validation or do I want to graph and see where overfitting starts to occur? Should have same effect.
# Decision tree CV. Max depth, min samples split, min samples leaf,
    """







