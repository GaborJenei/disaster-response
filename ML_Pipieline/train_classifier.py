import pickle
import sys
import time

import numpy as np
import pandas as pd
import sqlalchemy

import nltk

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from NLP_pipeline import tokenize, MessageLength

start_time = time.time()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(db_filepath):
    """
    Loads Data from SQL database

        Parameters
            database_filename(str): file path and file name of database

        Returns
            tuple of
             - x: numpy array of messages
             - y: numpy array of labels
             - category_names: list of str holding the category names
    """
    # load data from database
    engine = sqlalchemy.create_engine('sqlite:///' + db_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    x = df['message'].values

    category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
                      'search_and_rescue', 'security', 'military', 'water', 'food', 'shelter','clothing', 'money',
                      'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport',
                      'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                      'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
                      'direct_report']

    y = df[category_names].values

    return x, y, category_names


def build_model():
    """
    Build ML Pipeline and train GridSearchCV

        Parameters
            None

        Returns
            GridSearchCV object
    """

    # Pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, min_df=2)),
                ('tfidf', TfidfTransformer())
            ])),
            ('message_len', MessageLength())

        ])),
        ('clf', MultiOutputClassifier(SVC(), n_jobs=12)),
    ])

    # Parameters
    parameters = {
                  'clf__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'clf__estimator__C': [0.5, 1.0, 2.5, 5]
                  }

    pipeline_svm_cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', cv=3, verbose=4)

    return pipeline_svm_cv


def train(x, y, model):
    """
    Train Model with given x and y

        Parameters
            x: numpy array of features
            y: numpy array of labels
            model: Estimator (ML model, Pipeline, GridSearchCV) with a .fit method

        Returns
            model estimator object
    """
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # fit model
    model.fit(x_train, y_train)

    # output model test results
    return model


def evaluate_model(model, x_test, y_test, category_names):
    """
    Evaluates Estimator

        Parameters
            x_test: numpy array of test features
            y_test: numpy array of test set labels
            model: Estimator (ML model, Pipeline, GridSearchCV) with a .predict method
            category_names: list of string holding the category names

        Returns
            prints classification_report
            classification_report
    """

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=category_names)
    print(report)
    return report


def save_model(model, model_filepath):
    """
    Saves Estimator object into a python pickle file

        Parameters
            model: Estimator (ML model, Pipeline, GridSearchCV)
            model_filepath: str location to save the file to

        Returns
            None (saved pickle file)
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    save_model(model)  # save model


# if __name__ == '__main__':
#     data_file = sys.argv[1]  # get filename of dataset
#     run_pipeline(data_file)  # run data pipeline


X, y, categories = load_data('disaster_response.db')

# try to push it through 20% of the data?
sub_set = int(len(X)*0.1)
X_reduced = X[:sub_set]
y_reduced = y[:sub_set]

X_train, X_test, y_train, y_test = train_test_split(X, y)


pipeline_svm = build_model()
pipeline_svm.fit(X_train, y_train)
fit_time = time.time()

print('\n Pipeline GridSearch Fit time:')
print(fit_time - start_time)

print(y_train.size)
print(len(y_train))

print(pipeline_svm.best_estimator_.steps)

print(pipeline_svm.best_estimator_)
print('\n')
print(pipeline_svm.best_params_)
print('\n')
evaluate_model(pipeline_svm, X_test, y_test, categories)

save_model(pipeline_svm, 'saved_model.pkl')

end_time = time.time()
print(end_time-start_time)
# y_pred = pipeline_svm.predict(X_test)
