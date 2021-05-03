import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


def tokenize(text):

    genre = text.split()[-1]

    if genre=='social':
        tokenizer = TweetTokenizer(reduce_len=True, strip_handles=True)
    text = text.lower()

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    www_regex = 'www.*'

    text = re.sub(url_regex, "url_placeholder", text)
    text = re.sub(www_regex, "url_placeholder", text)
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)

    # tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    # Lemmatise - TODO add parameters to lemmatize verbs, nouns or verbs + nouns
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # lemmatize
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


class MessageLength(BaseEstimator, TransformerMixin):
    """
    Transformer taking a column of strings, and converting them into a column
    of integers, representing the word count of each string.
    """

    def word_count(self, text):
        # tokenize by words, how many words in message
        return len(tokenize(text))

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply word_count function to all values in X
        x_word_count = pd.Series(X).apply(self.word_count)
        return pd.DataFrame(x_word_count)
