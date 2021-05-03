import time

import numpy as np
import pandas as pd
import sqlalchemy

# NLP
import re
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import nltk
from sklearn.svm import LinearSVC, SVC

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from NLP_pipeline import tokenize, MessageLength

start_time = time.time()


def load_data(db_filepath):
    # load data from database
    engine = sqlalchemy.create_engine('sqlite:///' + db_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    x = (df['message'] + ', ' + df['genre']).values

    categories = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                  'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money',
                  'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport',
                  'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                  'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']

    y = df[categories].values

    return x, y, categories


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('message_len', MessageLength())

        ])),
        ('clf', MultiOutputClassifier(LinearSVC())),
    ])

    # Parameters
    # parameters = {
    #               'clf__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #               'clf__estimator__C': [0.5, 0.75, 1.0, 2.5],
    #               'clf__estimator__degree': [1, 2, 3, 4]
    #               }

    parameters = {
        'clf__estimator__C': [0.1],
    }

    pipeline_svm_cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', cv=3)

    return pipeline_svm_cv


# def train(X, y, model):
#     # train test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#     # fit model
#     model.fit(X_train, y_train)
#
#     # output model test results
#
#
#     return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    f1_scores = []
    for i, _ in enumerate(y_test):
        f1_scores.append(f1_score(y_test[:, i], y_pred[:, i], zero_division=1))

    min_f1 = np.min(f1_scores)
    avg_f1 = np.mean(f1_scores)
    max_f1 = np.max(f1_scores)

    print('Trained Model\n\tMin f1 score - {}\n\tAvg f1 score - {}\n\tMax f1 score - {}'.format(min_f1, avg_f1, max_f1))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

    pass


# def run_pipeline(data_file):
#     X, y = load_data(data_file)  # run ETL pipeline
#     model = build_model()  # build model pipeline
#     model = train(X, y, model)  # train model pipeline
#     export_model(model)  # save model
#
#
# if __name__ == '__main__':
#     data_file = sys.argv[1]  # get filename of dataset
#     run_pipeline(data_file)  # run data pipeline


X, y, categories = load_data('disaster_response.db')
X_train, X_test, y_train, y_test = train_test_split(X, y)

pipeline_svm = build_model()
pipeline_svm.fit(X_train, y_train)
evaluate_model(pipeline_svm, X_test, y_test)

end_time = time.time()
print(start_time - end_time)
# y_pred = pipeline_svm.predict(X_test)
