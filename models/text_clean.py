import re
import nltk
import swifter
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import (SnowballStemmer,
                       LancasterStemmer,
                       WordNetLemmatizer)

class CustomTextClean:
    """ Custom class to clean the text
    """
    def __init__(self, update_stopwords=True):
        self.lemmatizer = WordNetLemmatizer()
        self.snowball = SnowballStemmer(language='english')
        self.lancaster = LancasterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('English'))
        if update_stopwords:
            self.update_stopwords()
  
    def update_stopwords(self, words = ['no', 'nor', 'not']):
        for word in words:
            self.stopwords.remove(word)  

    def remove_contraptions(self, sentence):
        """ Removes the contraptions from the text
        Args:
            sentence(string):
                The text from which the contraptions are to
                be removed
        Returns:
            sentence(string):
                The final text without any contraptions
        """
        sentence = re.sub(r"won\'t", "will not", sentence)
        sentence = re.sub(r"can\'t", "can not", sentence)
        sentence = re.sub(r"shan\'t", "shall not", sentence)
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'s", " is", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'t", " not", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'m", " am", sentence)
        return sentence

    def clean(self, text, remove_contraptions=True):
        """ Performs the cleaning of the text by converting the
        text into lower case, then replace the contraptions, remove
        the punctions and finally remove any extra spaces from the
        text

        Args:
            text(string):
                The text which should be cleaned
            remove_contraptions(bool):
                Replace the contraptions with full forms if true
        Returns:
            text(string):
                The final cleaned text
        """
        text = text.lower()
        if remove_contraptions:
            text = self.remove_contraptions(text)
        text = re.sub(r'<[^>]*>', ' ', text)
        text = re.sub(r'[^\w\s]', ' ',text)
        # Remove any extra spaces
        text = re.sub(r' +',' ', text.strip())
        return text
    
    def plot_wordcloud(self, text, stop_words = None):
        """ Plots the wordcloud on the given text
        Args:
            text(string):
                The cleaned text
            stop_words(list):
                List of the stopwords that should not be considered
                while plotting the wordcloud
        """
        if not stop_words:
            stop_words = self.stopwords
        wordcloud = WordCloud(width = 1280,
                              height = 720,
                              background_color='white',
                              stopwords=stop_words).generate(text)
        # Set figure size
        fig = plt.figure(figsize=(7, 7))
        # Display image
        plt.imshow(wordcloud) 
        # No axis details
        plt.axis("off")

    def clean_text_data(self, summary_df, text_df):
        """ Combines the summary and text columns and then cleans the data

        Args:
            summary_df(DataFrame):
                A dataframe column consisting of summary of the review
            text_df(DataFrame):
                A dataframe column with complete text or description
                of review
        Returns:
            complete_df(DataFrame):
                A dataframe which consists of cleaned data with summary
                and text merged
        """
        complete_df = summary_df + " " + text_df
        complete_df.dropna(axis=0, how='any',inplace=True)
        complete_df['cleaned_text'] = complete_df.swifter.apply(lambda x: self.clean(x))
        return complete_df['cleaned_text']
    
    def clean_summary_text(self, text, summary=None):
        """ Combines the summary and text and then cleans the data

        Args:
            text(string):
                The description of single occurance
            summary(string):
                The summary of single occurance
        Returns:
            cleaned_text(string):
                The cleaned joint string of summary and text 
        """
        text = ' '.join([summary,text])
        cleaned_text = self.clean(text)
        return cleaned_text
    
    def sentence_lemmatization(self, sentence):
        """ Removes the stopwords and lemmatizes every other word
        in a sentence

        Args:
            sentence(string):
                The string data that should be used for lemmatization

        Returns:
            lemmatize_text(string):
                The final string with the lemmatized word
        """
        token_words = word_tokenize(sentence)
        lemmatize_text = ''
        for word in token_words:
            if word not in self.stopwords:
                word_lem = self.lemmatizer.lemmatize(word)
                lemmatize_text += word_lem + ' '
        return lemmatize_text.strip()
        
    def sentence_stemming(self, sentence, stem_model='snowball'):
        """ Removes the stopwords and lemmatizes every other word
        in a sentence

        Args:
            sentence(string):
                The string data that should be used for stemming
            stem_model(string):
                Applies snowball stemmer or lancaster stemmer

        Returns:
            lemmatize_text(string):
                The final string with the lemmatized word
        """
        if stem_model=='snowball':
            stem_model = self.snowball
        elif stem_model=='lancaster':
            stem_model = self.lancaster
        else:
            return None
        token_words = word_tokenize(sentence)
        stem_text = ''
        for word in token_words:
            if word not in self.stopwords:
                word_stem = stem_model.stem(word)
                stem_text += stem_text + word_stem + ' '
        return stem_text.strip()
            
    def perform_lemmatization(self, df):
        """ Wrapper function to apply lemmatization on the data frame

        Args:
            df(dataframe):
                The dataframe containing the data
        Returns:
            (dataframe)
        """
        return df.swifter.apply(lambda x: self.sentence_lemmatization(x))

    def perform_stemming(self, df,  stem_model='snowball'):
        """ Wrapper function to apply stemming on the data frame

        Args:
            df(dataframe):
                The dataframe containing the data
            stem_model(string):
                Applies snowball stemmer or lancaster stemmer
        Returns:
            (dataframe)
        """
        return df.swifter.apply(lambda x: self.sentence_stemming(x, stem_model))
