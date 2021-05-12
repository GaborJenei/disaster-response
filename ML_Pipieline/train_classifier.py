import pickle
import sys
import time

import pandas as pd
import sqlalchemy

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, make_scorer, f1_score

import nltk
from NLP_pipeline import tokenize, MessageLength

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
    df = pd.read_sql("""SELECT * FROM messages WHERE "message language"='en' """, engine)
    x = df['message'].values

    category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
                      'search_and_rescue', 'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money',
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

    # Grid search parameters tested
    # parameters = {
    #               'clf__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #               'clf__estimator__C': [0.5, 1.0, 2.5, 5]
    #               }

    # Best Parameters
    parameters = {
        'clf__estimator__kernel': ['linear'],
        'clf__estimator__C': [1.0]
    }

    # Create F1 scorer object with non-default parameters
    f1_scorer = make_scorer(f1_score, average='micro', zero_division=1)

    # GridSearch object to test different scenarios
    pipeline_svm_cv = GridSearchCV(pipeline, param_grid=parameters, scoring=f1_scorer, cv=3, verbose=4)

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

    y_pred = model.predict(x_test)
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


def main():
    if len(sys.argv) == 3:
        # measure training time
        start_time = time.time()

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        end_time = time.time()
        print('Trained model saved!\nTraining took {} minutes'.format((end_time - start_time) / 60))

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
