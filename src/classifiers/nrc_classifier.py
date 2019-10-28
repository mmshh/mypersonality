from src.util import Utils
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
import pickle

from src.util import Utils

class NRCClassifier:
    @staticmethod
    def generate_personality_trait_data(personality_trait):
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        nrc_df = util.read_data_to_dataframe("../../data/Train/Text/nrc.csv")
        nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
        merged_df = pd.merge(nrc_df, profile_df, on='userid')
        return merged_df.filter(['positive', 'negative', 'anger', 'anticipation', 'disgust',
       'fear', 'joy', 'sadness', 'surprise', 'trust', personality_trait], axis=1)

    @staticmethod
    def train(df, predicted_variable):
        X_train, X_test, y_train, y_test = Utils.split_data(df)
        regr = ElasticNet(random_state=0)
        regr.fit(X_train, y_train)
        # pickle.dump(regr, open("../resources/Elastic Net_" + predicted_variable + ".sav", 'wb'))
        y_pred = regr.predict(X_test)
        print("Elastic net acc: ", accuracy_score(y_test, y_pred))

        clf = linear_model.Lasso(alpha=0.1)
        clf.fit(X_train, y_train)
        # pickle.dump(clf, open("../resources/Lasso_" + predicted_variable + ".sav", 'wb'))
        y_pred = regr.predict(X_test)
        print("Lasso acc: ", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    df = NRCClassifier().generate_gender_data("ope")
    gooz =""
