import numpy as np
import pandas as pd
import sqlalchemy

# NLP
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import sklearn

# load data from database
engine = sqlalchemy.create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql("SELECT * FROM messages", engine)
X = df[['message', 'genre']]
Y = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
        'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
        'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools',
        'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire',
        'earthquake', 'cold', 'other_weather', 'direct_report']]


def tokenize(text):
    # TODO Clean commas, and all weird characters

    # TODO remove web sites, remove emails, deal with @mentions, replace with name?

    # TODO Get hashtags in a separate array?

    # tokenize text
    tokens = word_tokenize(text.lower())

    # initiate lemmatiser
    lemmed = []
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatise, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens
