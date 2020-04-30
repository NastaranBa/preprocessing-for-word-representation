# -*- coding: utf-8 -*-
"""

@author: Nastaran Babanejad
"""

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

News = pd.read_csv("C:.../input/News.csv")

text= News['content']
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

"""""
Making Vocabulary and Training the Model
(sg=0 CBOW , sg=1 Skip-gram)

"""""
#########

model1 = Word2Vec(sent2, min_count=5,size= 300,workers=multiprocessing.cpu_count(), window =1, sg = 0)
print('Done Training')

SizeOfVocab = model1.wv.vocab
print('Size of Vocabulary=',len(SizeOfVocab))
print('Done making the Vocabulary')

#####
model2 = Word2Vec(sent2, min_count=5,size= 300,workers=multiprocessing.cpu_count(), window =1, sg = 1)
print('Done Training')

SizeOfVocab = model2.wv.vocab
print('Size of Vocabulary=',len(SizeOfVocab))
print('Done making the Vocabulary')

""""
Saving the embeddings and the model

"""""

from gensim.models import Word2Vec, KeyedVectors   
model1.wv.save_word2vec_format('W-CBOW-Stem.bin.gz', binary=True)
model1.wv.save_word2vec_format('W-CBOW-Stem.txt', binary=False)
model1.save('W-CBOW-Stem.bin')
print('Done Saving Model1')
#####
model2.wv.save_word2vec_format('W-Skip-Stem.gz', binary=True)
model2.wv.save_word2vec_format('W-Skip-Stem.txt', binary=False)
model1.save('W-Skip-Stem.bin')
print('Done Saving Model2')

#model.save('model2.bin')

print('Done Saving the Embeddings')

