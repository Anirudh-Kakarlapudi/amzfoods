import os
import re
import nltk
import swifter
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.stem import (WordNetLemmatizer,
                       SnowballStemmer,
                       LancasterStemmer)

class CustomTextClean:
    """ Custom class to clean the text
    """
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
    
    def plot_wordcloud(self, text, stop_words=STOPWORDS):
        """ Plots the wordcloud on the given text
        Args:
            text(string):
                The cleaned text
            stop_words(list):
                List of the stopwords that should not be considered
                while plotting the wordcloud
        """
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
