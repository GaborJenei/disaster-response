import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


def tokenize(text):
    """
    Creates word tokens of input text

        Parameters
            text: str text to tokenize

        Returns
            tokenized text as a list of strings
    """
    text = text.lower()

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    www_regex = 'www.[\w\.]*'

    at_mention = '[@][\w]*'
    re_tweet = 'RT [\w]*'
    bit_ly = 'http bit\.ly [\w]*'
    tinyurl = 'http tinyurl\.com [\w]*'
    email = '[\w\.]*@[\w\.]*\.[\w]{0,4}'

    text = re.sub(url_regex, "url_placeholder", text)
    text = re.sub(www_regex, "url_placeholder", text)
    text = re.sub(bit_ly, "url_placeholder", text)
    text = re.sub(tinyurl, "url_placeholder", text)
    text = re.sub(email, "email", text)
    text = re.sub(re_tweet, "re_tweet", text)
    text = re.sub(at_mention, "at_mention", text)

    text = re.sub(r'[^a-zA-Z0-9]', " ", text)

    # tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = []
    for token in tokens:
        # lemmatize each tokens
        lemmatized_token = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemmatized_token)

    return lemmatized_tokens


class MessageLength(BaseEstimator, TransformerMixin):
    """
    Transformer taking a column of strings, and converting them into a column
    of integers, representing the word count of each string.
    """

    def word_count(self, text):
        """
        Counts the number of words in a text separated by white space

            Parameters
                text (str): text for counting the words

            Returns
                int: word count
        """
        # tokenize by words, how many words in message
        return len(tokenize(text))

    def fit(self, x, y=None):
        """
        Class is required to have a fit method to be able to use in a Pipeline

            Parameters
                x
                y

            Returns
                self
        """
        return self

    def transform(self, X):
        """
        Transforms array of strings to array of floats standing for the scaled  wordcount in all strings

            Parameters
                X: numpy array of strings to transform

            Returns
                column array of floats
        """
        # apply word_count function to all values in X
        x_word_count = pd.Series(X).apply(self.word_count)

        x_word_count_norm = x_word_count / x_word_count.max()

        return x_word_count_norm.values.reshape((-1, 1))
