import pickle
from collections import defaultdict

from sklearn import feature_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from src.util import Utils
import pandas as pd
import numpy as np


def generate_age_data():
    util = Utils()
    profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
    relation_df = util.read_data_to_dataframe("../../data/Train/Relation/Relation.csv")
    merged_df = pd.merge(relation_df, profile_df, on='userid')
    merged_df['age'] = pd.cut(merged_df['age'], [0, 25, 35, 50, 200], labels=["xx-24", "25-34", "35-49", "50-xx"],
                              right=False)
    return merged_df.filter(['userid', 'like_id', 'age'], axis=1)


def one_hot_encode(df, group_col, encode_col):
    grouped = df.groupby(group_col)[encode_col].apply(lambda lst: tuple((k, 1) for k in lst))
    category_dicts = [dict(tuples) for tuples in grouped]
    v = feature_extraction.DictVectorizer(sparse=False)
    X = v.fit_transform(category_dicts)
    one_hot = pd.DataFrame(X, columns=v.get_feature_names(), index=grouped.index)
    return one_hot


if __name__ == "__main__":
    data = generate_age_data()

    enc = OneHotEncoder(handle_unknown='ignore')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # gooss = X.groupby('like_id').count() > 5
    # gooss.rename(columns={'userid': 'chos'},inplace=True)

    # xnew = X.groupby('like_id').filter(lambda x: x['like_id'].min() > 5)
    # xnew.reset_index(drop=True, inplace=True)

    # X.pivot_table(index=['userid'], columns=['like_id'], aggfunc=[len], fill_value=0)

    # slightly better
    # one_hot = pd.get_dummies(X.iloc[:10], columns=['like_id'])
    # X = X.iloc[:1000]

    one_hot = one_hot_encode(X, 'userid', 'like_id')

    util = Utils()
    profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
    ids = one_hot.index.to_series().reset_index(drop=True)
    merged_df = pd.merge(ids, profile_df, on='userid')
    mdf = merged_df.filter(['neu'], axis=1)
    one_hot.reset_index(drop=True, inplace=True)
    mdf1000 = mdf.head(3000)
    one_hot1000 = one_hot.head(3000)
    # 3000 -> [0.58735441 0.58833333 0.58833333 0.58833333 0.58764608]
    # 1000 -> [0.58706468 0.58706468 0.59       0.59       0.59090909]

    # reg = LinearRegression()
    lasso = Lasso()

    # print(cross_val_score(lasso, one_hot1000, mdf1000, cv=3))

    # clf.fit(one_hot, mdf)
    # pickle.dump(clf, open("../resources/SGD_Age1.sav", 'wb'))
    X_train, X_test, y_train, y_test = train_test_split(one_hot1000, mdf1000, test_size=0.33, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    gooz = ""