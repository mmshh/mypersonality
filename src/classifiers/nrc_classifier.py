from src.util import Utils
import pandas as pd

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


if __name__ == "__main__":
    df = NRCClassifier().generate_gender_data("ope")
    gooz =""