import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    This method will read and load database_filepath our data
    which is disaster_response_df
    
    
    Parametrs: database_filepath:string
    
    Return  test, train, category_names
    
    """
  
    df = pd.read_sql_table('messages', create_engine('sqlite:///{}'.format(database_filepath))) 
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
   """
    This method will read text and tokenize it
    
    Parametrs:
    text:string
    
    Return clean_tokens(which is the text after tokenizing it
    
    """
    
    #normlize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenize text
    tokens = word_tokenize(text)
    #remove stopwords
    stop_w = stopwords.words("english")
    tokens = [token for token in tokens if token not in stop_w ]
    
    lemmatizer = WordNetLemmatizer()
    # lemmatizer text
    clean_tokens = []
    for t in tokens:
        clean_token = lemmatizer.lemmatize(t).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
      """
    This method will build a model and return it
    
    Parametrs:
    notheng
    
    Return cv(the model that we created)
    
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [9, 24],
                  'tfidf__use_idf': [True, False]}
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
      """
    This method will evaluate the model
    
    Parametrs:
    model, X_test, Y_test, category_names
    
    Return void
    
    """
    y_predicted = model.predict(X_test)
    print("Evaluation Report:\n")
    
    i=0
    for column in Y_test:
        #Print column name
        print(column) 
        #Printing classification report for every class
        print(classification_report(Y_test[column], y_predicted[:,i]))
        i=i+1


def save_model(model, model_filepath):
     """
    This method will save the model
    
    Parametrs:
    model, model_filepath
    
    Return void
    
    """
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)





def main():
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