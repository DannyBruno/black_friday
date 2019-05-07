"""

Initial data analysis.

"""

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.style.use('seaborn')

from dataset import load_data


def univariate_analysis(df):
    """
    Univariate.
    """

    # Purchase amounts.
    sns.distplot(df['Purchase'], bins=25)
    plt.xlabel("Purchase Amount")
    plt.ylabel("Number of Buyers")
    plt.title("Purchase Distribution")
    plt.savefig('plots/purchase_amounts.png')

    # How many occupations?
    sns.countplot(df['Occupation'])
    plt.xlabel("Occupation")
    plt.ylabel("Count")
    plt.title("Occupation Distribution")
    plt.savefig('plots/occupation_dist.png')

    # Distribution of marital status.
    sns.countplot(df['Marital_Status'])
    plt.xlabel("Marital Status")
    plt.ylabel("Count")
    plt.title("Marital Status Distribution")
    plt.savefig('plots/marital_status.png')

    # Distribution of product category 1-3.
    sns.countplot(df['Product_Category_1'])
    plt.xlabel("Product Category")
    plt.ylabel("Count")
    plt.title("Product Category 1 Dist.")
    plt.savefig('plots/product_cat_1_dist.png')

    sns.countplot(df['Product_Category_2'])
    plt.xlabel("Product Category")
    plt.ylabel("Count")
    plt.title("Product Category 2 Dist.")
    plt.savefig('plots/product_cat_2_dist.png')

    sns.countplot(df['Product_Category_3'])
    plt.xlabel("Product Category")
    plt.ylabel("Count")
    plt.title("Product Category 3 Dist.")
    plt.savefig('plots/product_cat_3_dist.png')



def correlation(df):
    numeric_features = df.select_dtypes(include=[np.number])

    corr = numeric_features.corr()

    print(corr['Purchase'].sort_values(ascending=False)[:10])

    sns.heatmap(corr, vmax=.8, annot_kws={'size': 20}, annot=True)
    #plt.savefig('plots/corr_heatmap.png')
    plt.show()


def bivariate_analysis(df):

    # Occupation and purchase amount.
    occupation_pivot = df.pivot_table(index='Occupation', values='Purchase', aggfunc=np.mean)
    occupation_pivot.plot(kind='bar', color='blue', figsize=(12, 7))
    plt.xlabel("Occupation")
    plt.ylabel("Avg. Purchase")
    plt.title("Occupation v. Purchase")
    #plt.savefig('plots/occupation_v_purchase.png')
    plt.show()

    # Distribution of purchase amount for each occupation. Seems like a weak predictor b/c they all seem to share the same distribution/similar mean.
    occ_purch_dist = df.loc[(df['Occupation'] == 0)]
    occ_purch_dist.hist('Purchase', bins=25)
    plt.xlabel("Purchase Value")
    plt.ylabel("Frequency")
    plt.title("Occupation 0 Purchase Distribution")
    #plt.savefig('plots/occup_0_purchase_dist.png')
    plt.show()

    # Marital Status and purchase amount. Again no correlation.
    marital_pivot = df.pivot_table(index='Marital_Status', values='Purchase', aggfunc=np.mean)
    marital_pivot.plot(kind='bar', color='blue', figsize=(12, 7))
    plt.xlabel("Marital Status")
    plt.ylabel("Avg. Purchase")
    plt.title("Marital Status v. Purchase")
    #plt.savefig('plots/marital_v_purchase.png')
    plt.show()

    # Product categories and purchase amount. Correlation that differs from frequency of purchases in those areas.
    pcat_pivot = df.pivot_table(index='Product_Category_1', values='Purchase', aggfunc=np.mean)
    pcat_pivot.plot(kind='bar', color='blue', figsize=(12, 7))
    plt.xlabel("Product Category 1")
    plt.ylabel("Avg. Purchase")
    plt.title("Produt Category 1 v. Purchase")
    plt.savefig('plots/pcat1_v_purchase.png')
    plt.show()






if __name__ == "__main__":
    df = load_data()
    #univariate_analysis(df)
    #correlation(df)
    bivariate_analysis(df)
