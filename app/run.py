import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Table
# from sklearn.externals import joblib
from sqlalchemy import create_engine

import pickle


app = Flask(__name__)

def tokenize(text):
    """ Tokenizes the imput text and performs basic text cleaning"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = pickle.load(open("../models/cls_nogrid.pkl", 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    cat_names = df.columns[4:]
    cat_sums = df.sum()[cat_names]
    total = sum(cat_sums)
    small_sums = [cat for cat in cat_sums.values if cat < 0.1*total]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(labels=cat_names, values=cat_sums, insidetextfont={'size': 1})
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(labels=genre_names, values=genre_counts)
            ],
            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },
        {
            'data': [
                Table(
                    header=dict(values=['id', 'message', 'genre']),
                    cells=dict(values=[df.id, df.message, df.genre])
                )
            ],
            'layout': {
                'title': 'Messages Table'
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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    # app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(host='127.0.0.1', port=3001, debug=True)

if __name__ == '__main__':
    main()