import numpy as np
import pandas as pd
import sqlalchemy

# NLP
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# load data from database
engine = sqlalchemy.create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql("SELECT * FROM messages", engine)
X = df['message'].values
Y = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
        'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
        'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools',
        'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire',
        'earthquake', 'cold', 'other_weather', 'direct_report']].values


# Todo add parameters so we can use them through GridSearch
# Todo add tokenize to Pipeline
def tokenize(text):
    # TODO normalise: Clean commas, and all weird charachters
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    mention_regex = '/^(?!.*\bRT\b)(?:.+\s)?@\w+/i'

    text = text.lower()

    text = re.sub(url_regex, "url_placeholder", text)
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    # TODO nltk tweet tokenizer remove web sites, remove emails, deal with @mentions, replace with name?

    # Get hashtags in a separate array?

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


# Pipeline
# TODO Try different Classifiers: RandomForest, KNN, etc
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier(n_jobs=12))),
])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, Y)

# train classifier
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
