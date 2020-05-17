import sys
from sqlalchemy import create_engine
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


def load_data(database_filepath):
    """
    Loads data from database into dataframes for X and Y

    Parameters:
    database_filepath (str): the filepath to the database

    Returns:
    X: a dataframe of the messages
    Y: a dataframe of the categories (indexed same as X)
    category_names: list of category names from Y
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """ Tokenizes the imput text and performs basic text cleaning"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Creates a sklearn model pipeline and gridsearch parameters

    Returns:
    GridSearchCV: a GridSearchCV object to use for fitting and predicting
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('cls', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'cls__estimator__n_estimators': [50, 100, 200],
        'cls__estimator__min_samples_split': [2, 3, 4],
    }
    
    # Verbose to display messages during training (default None)
    # n_jobs to run training jobs in parallel (default 1)
    # cv to limit the crossvalidation (default 5)
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of the ML model

    Parameters:
    model (Pipeline): a Pipeline or GridSearchCV object to predict with
    X_test: the message df to make predictions about
    Y_test: the categories with which to judge the results of the model
    category_names: the names of each category (column) in Y_test

    """

    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        pred = [p[i] for p in y_pred] 
        true = Y_test[col].values
        out = classification_report(true, pred)
        print('{}'.format(col))
        print(out)
    pass


def save_model(model, model_filepath):
    """ a pickle object that stores the trained ML model at model_filepath"""
    s = pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Performs an ML pipeline on the input data. sqllite database to trained model

    Parameters:
    database_filepath (str): the filepath to the database
    model_filepath (str): the filepath to store the trained model
    """

    if len(sys.argv) == 3:
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

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()