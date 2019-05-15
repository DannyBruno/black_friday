"""

Attempting interpolation of product category.

"""
import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import model_selection, metrics



def vizualize_product_cat(df):
    '''

    Attempt to visualize in point cloud.

    '''

    # See single category distributions in analysis.py

    # Two valid categories - Pairs.
    cat_one_two = df[(df['Product_Category_1'] != -2) & (df['Product_Category_2'] != -2)]
    cat_one_three = df[(df['Product_Category_1'] != -2) & (df['Product_Category_3'] != -2)]
    cat_two_three = df[(df['Product_Category_2'] != -2) & (df['Product_Category_3'] != -2)]

    plt.scatter(cat_one_two['Product_Category_1'][:5000], cat_one_two['Product_Category_2'][:5000])
    plt.title('Category 1 vs. Category 2')
    plt.xlabel('Category 1')
    plt.ylabel('Category 2')
    plt.savefig('plots/knn_one_two.png')
    plt.show()


    plt.scatter(cat_one_three['Product_Category_1'][:5000], cat_one_three['Product_Category_3'][:5000])
    plt.title('Category 1 vs. Category 3')
    plt.xlabel('Category 1')
    plt.ylabel('Category 3')
    plt.savefig('plots/knn_one_three.png')
    plt.show()

    plt.scatter(cat_two_three['Product_Category_2'][:5000], cat_two_three['Product_Category_3'][:5000])
    plt.title('Category 2 vs. Category 3')
    plt.xlabel('Category 2')
    plt.ylabel('Category 3')
    plt.savefig('plots/knn_two_three.png')
    plt.show()



    # Three valid categories - Triples.

    cat_all_three = df[(df['Product_Category_1'] != -2) & (df['Product_Category_2'] != -2) & (df['Product_Category_3'] != -2)]

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(cat_all_three['Product_Category_1'][:5000], cat_all_three['Product_Category_2'][:5000], cat_all_three['Product_Category_3'][:5000])
    plt.title('Category 1 vs. Category 2 vs. Category 3')
    ax.set_xlabel('Category 1')
    ax.set_ylabel('Category 2')
    ax.set_zlabel('Category 3')
    plt.savefig('plots/knn_3d.png')
    plt.show()


def interpolation_model_fit(df, tree_two, tree_three):
    '''
    Takes in two decision trees. Trains each one to predict their respective field (Product Category 2 or Product Category 3).
    Bad code duplication from model.py but oh well.
    '''

    # Select entries with all 3 as training/testing data.
    data = df[
        (df['Product_Category_1'] != -2) & (df['Product_Category_2'] != -2) & (df['Product_Category_3'] != -2)]

    features = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']

    for target in ['Product_Category_2', 'Product_Category_3']:

        print(f'Training for {target}!')

        if target == 'Product_Category_2':
            alg = tree_two
        else:
            alg = tree_three

        # Select features.
        features.remove(target)

        # Split.
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            data[features], data[target],
            test_size=.2)

        alg.fit(X_train, y_train)
        train_predictions = alg.predict(X_train)
        print(f'Training Accuracy: {metrics.accuracy_score(y_train, train_predictions)}')

        test_predictions = alg.predict(X_test)
        print(f'Training Accuracy: {metrics.accuracy_score(y_test, test_predictions)}')

        features.append(target)

    return tree_two, tree_three


def interpolation_tree(row, tree_two, tree_three):

    if row['Product_Category_2'] == -2 and row['Product_Category_3'] != -2:
        row['Product_Category_2'] = tree_two.predict([[row['Product_Category_1'], row['Product_Category_3']]])[0] # Hopefully this is an np.array
    elif row['Product_Category_2'] != -2 and row['Product_Category_3'] == -2:
        row['Product_Category_3'] = tree_three.predict([[row['Product_Category_1'], row['Product_Category_2']]])[0]

    return row[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']]


def interpolation_mode_setup(df, modes_two, modes_three):

    # Select those with all 3 entries.
    data = df[
        (df['Product_Category_1'] != -2) & (df['Product_Category_2'] != -2) & (df['Product_Category_3'] != -2)]

    # Loop through, add to appropriate maps if doesn't exist already.
    for idx, row in df.iterrows():

        if (row['Product_Category_1'], row['Product_Category_3']) not in modes_two.keys():
            # If doesn't exists already select all rows that match in 1 and 3. Find the Mode of 2 in those rows.
            modes_two[(row['Product_Category_1'], row['Product_Category_3'])] = mode(
                df[(df['Product_Category_1'] == row['Product_Category_1'])
                   & (df['Product_Category_3'] == row['Product_Category_3'])]['Product_Category_2']
            )[0][0]
        elif (row['Product_Category_1'], row['Product_Category_2']) not in modes_three.keys():
            modes_three[(row['Product_Category_1'], row['Product_Category_2'])] = mode(
                df[(df['Product_Category_1'] == row['Product_Category_1'])
                   & (df['Product_Category_2'] == row['Product_Category_2'])]['Product_Category_3']
            )[0][0]

    return modes_two, modes_three


def interpolation_mode(row, modes_two, modes_three):
    '''
    TODO: Compute missing value by taking mode in column with each.
    Given product categories 1 & 2 predict the 3rd by selecting the entries that include values in all 3 fields,
    and that match in category 1 & 2. Then assigns 3 to the mode of category 3 for those entries.
    Modes stores a hash table of [cat1, cat2] -> mode cat3 in entries that match on cat1 and cat2.
    '''

    if row['Product_Category_2'] == -2 and row['Product_Category_3'] != -2 and\
            (row['Product_Category_1'], row['Product_Category_3']) in modes_two.keys():
       row['Product_Category_2'] = modes_two[(row['Product_Category_1'], row['Product_Category_3'])]
    elif row['Product_Category_2'] != -2 and row['Product_Category_3'] == -2 and\
            (row['Product_Category_1'], row['Product_Category_2']) in modes_three.keys():
        row['Product_Category_3'] = modes_three[(row['Product_Category_1'], row['Product_Category_2'])]

    return row[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']]


def impute_missing_vals(method, row, modes, tree_two, tree_three):
    """
    Potentially use to select which methods we want. Will most likely depricate.
    """
    # Try several methods for imputation.
    if method == 0:
        return interpolation_mode(row, modes)

    # Method 2: Predict using dt.
    elif method == 1:
        return interpolation_tree(row, tree_two, tree_three)





if __name__ == '__main__':
    df = pd.read_csv("data/BlackFriday_Modified.csv")
    vizualize_product_cat(df)   # Visualization shows promising results for predictability. We see roughly 58 discrete clusters for 1000 samples in the 3D plot.

    # Test 2 methods, modes and simple decision tree.




