import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

import NLP_pipeline

app = Flask(__name__)

# Instantiate the tokenizer
tokenize = NLP_pipeline.tokenize

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../ML_Pipieline/saved_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Bar charts of languages
    messages_by_language = df['message language'].value_counts()
    language_counts = list(messages_by_language.values)
    languages = list(messages_by_language.index)

    categories = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                  'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
                  'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools',
                  'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm',
                  'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']

    count_by_category = df[categories].sum()

    # Heatmap data prep
    count_by_cat_genre = df.groupby(by=['genre'])[categories].sum()

    # create visuals
    graphs = [
        # Bar chart by language
        {
            'data': [Bar(x=languages, y=language_counts)],
            'layout': {'title': 'Distribution of Message Languages',
                       'yaxis': {'title': "Count", 'type': "log"},
                       'xaxis': {'title': "Language"}
                       }
        },

        # Bar chart by category
        {
            'data': [Bar(x=count_by_category.index, y=count_by_category.values)],
            'layout': {'title': 'Distribution of Message Categories',
                       'yaxis': {'title': "Count", 'type': "log"},
                       'xaxis': {'title': "Category"}
                       }
        },

        # Bar chart by category
        {
            'data': [Heatmap(z=count_by_cat_genre.values,
                             x=count_by_cat_genre.columns,
                             y=count_by_cat_genre.index)],
            'layout': {'title': 'Distribution of Message Categories by Genre',
                       'yaxis': {'title': "Genre"},
                       'xaxis': {'title': "Category"}
                       }
        },

        # Bar chart by genre
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    print(query)

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    print(classification_labels)

    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
