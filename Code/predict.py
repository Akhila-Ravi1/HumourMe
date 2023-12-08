import pandas as pd
import numpy as np
import nltk
import re
from keras.models import load_model

# Import the word2vec model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api

class PredictJoke:
    def __init__(self):
        self.model = load_model('Models/LSTM_pretrained.keras')
        self.wv_model = api.load('word2vec-google-news-300')

    def preprocess_text(self, data):
        # convert to lower case
        clean_text = data.lower()
        
        #remove hyphens and sub with space
        clean_text = clean_text.replace("-"," ")
        clean_text = clean_text.replace(":"," ")
        clean_text = clean_text.replace("..."," ")

        # tokenize the data
        clean_text = clean_text.split(" ")

        # remove non alphanumeric chars
        clean_text = " ".join([re.sub(r'[^a-zA-Z0-9]', '', word) for word in clean_text])

        # remove stopwords
        stopword = nltk.corpus.stopwords.words('english')
        text_cleaned = " ".join([word for word in re.split('\W+', clean_text) if word not in stopword])

        # perform lemmatizing
        wn = nltk.WordNetLemmatizer()
        text_cleaned = " ".join([wn.lemmatize(word,'v') for word in re.split('\W+', text_cleaned)])

        return text_cleaned 

    def get_word2vec(self, data):
        doc_vector = []

        # preprocess the data
        data = self.preprocess_text(data)
        print(data)
        
        for word in data.split():
            try:
                # get the word vectors
                vc = self.wv_model[word]
                # append the word vectors to the list
                doc_vector.append(vc)
            except KeyError:
                continue
        
        doc_vector = np.array(doc_vector)

        # pad the sequence
        doc_vector = pad_sequences([doc_vector], maxlen=200, dtype='float32', value=[0] * doc_vector[0].shape[-1])

        return doc_vector

    def predict_joke(self, data, threshold=0.5):
        print(data)
        # get the word2vec representation of the input
        doc_vector = self.get_word2vec(data)

        # predict the input
        prediction = self.model.predict(doc_vector)
        print(prediction)

        if prediction[0][0] > threshold:
            prediction = "Humorous"
        else:
            prediction = "Not Humorous"

        return prediction

if __name__ == '__main__':
    predict = PredictJoke()
    user_input = input("Enter your joke here:")
    pred = predict.predict_joke(user_input)
    print(user_input)
