from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from keras.models import load_model
import spacy
from eventregistry import *
er = EventRegistry(apiKey='5014cfa8-0ccd-4e15-8e6d-6b749fd99a64')

nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)
label_dict = {0:'Agree',1:'Disagree',2:'Discuss',3:'Unrelated'}
model = load_model('keras_model_updated')
model._make_predict_function()
count_vectorizer = pickle.load(open('count_vectorizer1.pk','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pk','rb'))
#print(count_vectorizer)
#print(tfidf_vectorizer)
#print(model.summary())

def calculate_tfidf_score(headline,body):
    '''Calculates cosine similarity between the headline and body. '''

    headline_features = tfidf_vectorizer.transform([headline]).toarray()
    body_features = tfidf_vectorizer.transform([body]).toarray()
    cosine_similarity  = np.dot(headline_features,body_features.T)

    return cosine_similarity[0][0]

def create_input_for_model(headline,body,tfidf_score):
    '''Creates input for the model.'''

    headline_features = count_vectorizer.transform([headline]).toarray()
    body_features = count_vectorizer.transform([body]).toarray()
    input_feature = np.column_stack((headline_features,tfidf_score,body_features))

    return input_feature[0]


@app.route('/predict',methods=['POST'])
def predict():
    '''Predicts stance for given headline and body.'''

    headline = request.get_json()['headline']
    body = request.get_json()['body']
    print(headline)
    print(body)
    tfidf_score = calculate_tfidf_score(headline=headline,body=body)
    print(tfidf_score)
    inputs = create_input_for_model(headline=headline,body=body,tfidf_score=tfidf_score)
    model.summary()
    inputs = inputs.reshape((1,-1))
    prediction = model.predict(inputs)
    output = np.argmax(prediction)
    stance = label_dict[output]
    response = {'Stance':stance}

    return jsonify(response)


def extract_keywords(doc):
    '''A very simple method of extracting keywords from headline based on Named-Entity Recognition and Parts-of-Speech Tagging.'''

    doc = nlp(doc)

    nouns = []
    for token in doc:
        if token.pos_ in ['NNS','NN']:
            nouns.append(token.text)

    ents = []
    if doc.ents:
        for ent in doc.ents:
            ents.append(ent.text)
            #print(f'{ent.text:{20}}{ent.label_:{10}}{spacy.explain(ent.label_):{40}}{ent.start:{20}}{ent.end:{20}}')
    else:
        print('No entities')
        pass

    keywords = ents + nouns
    return keywords

@app.route('/sources',methods=['POST'])
def fetch_similar_news():
    ''' Fetch similar news based on keywords extracted from the headline. Works as an information retrieval system.'''

    headline = request.get_json()['headline']
    keywords = extract_keywords(headline)
    query = QueryArticles(keywords= QueryItems.AND(keywords))
    query.setRequestedResult(RequestArticlesInfo(count=5))
    response = er.execQuery(query)
    articles = response['articles']
    results = articles['results']
    predictions = []
    for result in results:
        try:
            body = result['body']
            url = result['url']
            title = result['title']
            # print(url)
            # print(title)
            pred = predict_stance_for_sources(headline=headline,body=body)
            predictions.append(pred)
        except KeyError as e:
            pass

    return jsonify(predictions)


def predict_stance_for_sources(headline,body):
    '''Predict stance for events fetched from other sources similar to the given headline.'''

    tfidf_score = calculate_tfidf_score(headline=headline,body=body)
    input = create_input_for_model(headline=headline,body=body,tfidf_score=tfidf_score)

    input = input.reshape((1,-1))
    prediction = model.predict(input)
    output = np.argmax(prediction)
    stance = label_dict[output]
    response = {'Stance':stance}

    return response



if __name__ == '__main__':
    app.run(port=5000,debug=True)
