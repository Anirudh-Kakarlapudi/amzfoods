import re
import nltk
import swifter
import numpy as np
import pandas as pd
from text_clean import CustomTextClean
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

class PreProcess(CustomTextClean):
    """ Has the functions to process the data - 
    clean the data and save the final data into
    the csv
    """
    def df_clean(self, df, lemmatize=True, stemming = False, stem_model='snowball'):
        """ Cleans the amazon fine food reivews dataset

        Args:
            df(dataframe):
                The dataframe consisiting of reviews
            lemmatize(boolean):
                Peform lemmatization on the text from reviews
            stemming(boolean):
                Performs the stemming process on the text data
                if lemmatize is not true.
            stem_model(string):
                The type of stemmer to be used
                'snowball' -> SnowballStemmer
                'lancaster' -> LancasterStemmer
        Returns:
            
        """
        cleaned_df = pd.DataFrame()
        df.dropna(axis=0, how='any', inplace=True) 
        no_duplicates_df = df.drop_duplicates(subset=['ProductId','UserId', 'Time'])
        no_illegals_df = no_duplicates_df[no_duplicates_df.HelpfulnessNumerator <= 
                                            no_duplicates_df.HelpfulnessDenominator]
        no_illegals_df['cleaned_text'] = self.clean_text_data(no_illegals_df['Summary'],
                                                              no_illegals_df['Text'])
        cleaned_df['score'] = no_illegals_df['Score'].apply(lambda x:1 if x>3 else 0)

        if lemmatize:
            cleaned_df['cleaned_text'] = self.perform_lemmatization(no_illegals_df['cleaned_text'])
        elif stemming and stem_model in {'snowball','lancaster'}:
            cleaned_df['cleaned_text'] = self.perform_stemming(no_illegals_df['cleaned_text'],
                                                               stem_model)
        else:
            cleaned_df['cleaned_text'] = no_illegals_df['cleaned_text']
        return cleaned_df
    
    def apply_split(self, df, test_size=0.2):
        """ Splits the data into train and test dataframe based on test
        size

        Args:
            df(dataframe)
            test_size(float):
                The size of the test dataframe
        Returns:
            (list)
                The list with train and test dataframe
        """
        train, test = train_test_split(df,
                                       test_size=test_size,
                                       random_state=42,
                                       shuffle=False)
        return (train, test)

    def save_data(self, train, test):
        """ Saves the train and test dataframes into the csv files

        Args:
            train(dataframe)
            test(dataframe)
        """
        train.to_csv('data/train.csv')
        test.to_csv('data/test.csv')


if __name__ == "__main__":
    pp = PreProcess()
    df = pd.read_csv('../data/Reviews.csv')
    cleaned_df = pp.df_clean(df)
    train_df, test_df = pp.apply_split(cleaned_df, 0.33)
    pp.save_data(train_df, test_df)
