# -*- coding: utf-8 -*-
"""

@author: Nastaran Babanejad
"""
import argparse
import csv
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from keras.layers import Input, Embedding, LSTM, Dense
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

import csv
import nltk
import re
from gensim.models import Word2Vec
import pandas as pd 
import numpy as np
from sklearn.datasets import load_files
from collections import Counter 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.porter import PorterStemmer
import gc

movie_reviews = pd.read_csv(".../input/IMDB_Dataset.csv")


text= movie_reviews['review']
X=text.values.tolist()

def majid(X):
    corpus = []
    for i in range(0, len(X)):
        #review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(X[i])) #remove punctuation
        review = re.sub(r'\d+',' ', str(X[i]))# remove number
        review = review.lower() #lower case
        review = re.sub(r'\s+', ' ', review) #remove extra space
        review = re.sub(r'<[^>]+>','',review) #remove Html tags
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.append(review)        
#    return corpus        
    #Tokenizing and Word Count  
    words=[]
    for i in range(len(corpus)):
        words= nltk.word_tokenize(corpus[i])
        #sentences.append(words)
   
    return words

X = [[el] for el in X] 

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
sentences = Parallel(n_jobs=num_cores)(delayed(majid)(i) for i in X)

'''
#############
#Porter stemmer
#############
def stemming(sentences):
        porter = PorterStemmer()
        #words = sent.split() 
        stemmed_words = [porter.stem(word) for word in sentences]
        return stemmed_words

def majid2(X):
    sentences= stemming(X)
    gc.collect()
    return sentences
 
from joblib import Parallel, delayed
import multiprocessing

#X = [[el] for el in X]
    
num_cores = multiprocessing.cpu_count()
sent2 = Parallel(n_jobs=num_cores)(delayed(majid2)(i) for i in sentences)
'''
#######################
#Snowball Stemmer
########################
from nltk.stem.snowball import SnowballStemmer

import nltk
def stemming2(sentences):
        sno = nltk.stem.SnowballStemmer('english')
        #words = sent.split() 
        stemmed_words = [sno.stem(word) for word in sentences]
        return stemmed_words

def majid2(X):
    sentences= stemming2(X)
    gc.collect()
    return sentences
 
from joblib import Parallel, delayed
import multiprocessing

#X = [[el] for el in X]
    
num_cores = multiprocessing.cpu_count()
sent2 = Parallel(n_jobs=num_cores)(delayed(majid2)(i) for i in sentences)
###############################
###############################
X = [' '.join(x) for x in sent2]
#sent4= TreebankWordDetokenizer().detokenize([ i for i in [sent3]])

#after preprocess our reviews we will store them in a new list 
    
#we need to convert our labels into digits    
y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
#Y = pd.get_dummies(df['Product']).values
#print('Shape of label tensor:', Y.shape)

#We  use train_test_split method from the sklearn.model.selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


#Preparing the Embedding Layer
#use the Tokenizer class from the keras.preprocessing.
#text module to create a word-to-index dictionary.
#In the word-to-index dictionary, each word in the corpus is used as a key, 
#while a corresponding unique index is used as the value for the key
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
print('Vocab size:',vocab_size)
maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#We use pretrained embeddings to create our feature matrix. 
#In the following script we load the pretrained word embeddings 
#and create a dictionary that will contain words as keys and 
#their corresponding embedding list as values.
from numpy import array
from numpy import asarray
from numpy import zeros
from gensim.models import Word2Vec

embeddings_dictionary = dict()
glove_file = open('.../Embeddings.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:])
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

#we create an embedding matrix where each row number 
#will correspond to the index of the word in the corpus.
#The matrix will have 300 columns where each column 
#will contain the pretrained word embeddings for the words in our corpus.
embedding_matrix = zeros((vocab_size, 300))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

        

model = Sequential()
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))

#########################
# define and fit the model
def get_model(X_train, y_train):
	# define model
	#model = Sequential()
	#model.add(Dense(100, input_dim=2, activation='relu'))
	#model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	# fit model
	model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.10)
	return model

model = get_model(X_train, y_train)
####################

''''
from keras import backend as K

def recall_m(y_train, y_test):
        true_positives = K.sum(K.round(K.clip(y_train * y_test, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_train, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_train, y_test):
        true_positives = K.sum(K.round(K.clip(y_train * y_test, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_train, y_test):
    precision = precision_m(y_train, y_test)
    recall = recall_m(y_train, y_test)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
'''
'''
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_test, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
####
    '''
# compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# fit the model
#history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.10)

############
#from sklearn.metrics import classification_report
#from sklearn.metrics import precision_recall_fscore_support
#y_pred = model.predict(X_test)
#print(precision_recall_fscore_support(y_test, y_pred))

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
yhat_probs = model.predict(X_test, verbose=1)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=1)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)

#print(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
#print(classification_report(y_test, y_pred, average='weighted'))


# evaluate the model
#loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=1)
####################

#score = model.evaluate(X_test, y_test, verbose=1)
#print("Test Score:", score[0])
#print("Test Accuracy:", score[1])
#y_true = y_test
#target_names = ['1','0']
#print(classification_report(y_true, y_pred, target_names=target_names))



