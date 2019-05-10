'''

Model framework.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics

# Models.
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

target = 'Purchase'
IDcol = ['User_ID', 'Product_ID']


def modelfit(alg, data, features, target):

    # Split data.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data[features], data[target],
                                       test_size=.3)

    print(str(X_train.shape))
    # Fit model.
    alg.fit(X_train, y_train)

    # Predict test data.
    train_predictions = alg.predict(X_train)

    # Assess performance on CV.
    cv_score = model_selection.cross_val_score(alg, X_train, y_train, cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    # Display report.
    print("Model Report:")
    print("RMSE: %.4g" % np.sqrt(metrics.mean_squared_error(y_train, train_predictions)))
    print("CV Score: Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Predict on test data.
    test_predictions = alg.predict(X_test)



if __name__ == '__main__':

    # Read in data.
    df = pd.read_csv("data/BlackFriday_Modified.csv", nrows=50000)
    features = df.columns.drop(['Purchase', 'Product_ID', 'User_ID'])


    # Linear Regression Model. (Similar to regressions.py)
    print("Linear Regression Model")
    LR = LinearRegression(normalize=True)
    modelfit(LR, df, features, target)

    # What is contributing?
    coef1 = pd.Series(LR.coef_, features).sort_values()
    coef1.plot(kind='bar', title='LR Coeffs')
    plt.show()

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

    # Decision Tree Regressor. (Now testing more complex models.) (These all need some hyperparameter tuning.)
    print("Decision Tree Regressor")
    DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
    modelfit(DT, df, features, target)

    # Random Forest Regressor.
    print("Random Forrest Regressor")
    RF = RandomForestRegressor(n_estimators=20, max_depth=15)
    modelfit(RF, df, features, target)










