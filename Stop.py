# -*- coding: utf-8 -*-
"""
@author: Nastaran Babanejad

"""
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.porter import PorterStemmer
import csv
import nltk
import re
from gensim.models import Word2Vec
import pandas as pd 
import numpy as np
from sklearn.datasets import load_files
from collections import Counter 
import gc

News = pd.read_csv("C:.../input/WikiArticles.csv", names=['section_text'])
print('Done Importing')

"""
Pre-processing Text
 (for the Baseline we only remove the extra Whitespace in this step)
 
"""

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
#############

def remove_stopwords(sentences):
        stopwords_list = stopwords.words('english')
#        words = sentences.split() 
        clean_words = [word for word in sentences if (word not in stopwords_list)] 
        return clean_words 

#sentences= remove_stopwords(sent)

def majid2(X):
    sentences= remove_stopwords(X)
    gc.collect()
    return sentences
 
from joblib import Parallel, delayed
import multiprocessing

#X = [[el] for el in X]
    
num_cores = multiprocessing.cpu_count()
sent2 = Parallel(n_jobs=num_cores)(delayed(majid2)(i) for i in sentences)

#print(len(stopwords.words('english')))

count = 0
c = {}
for words in sent2:
  for s in words:
    if s in c:
        c[s] += 1
    else:
        c[s] = 1
    count+=1
    #if (word_counter % 10000)  == 0:
    #    print(word_counter)
d= []

for k,v in c.items():
    if v == 1:
        d.append(k) 
    
print('Corpus Size=', count)        
print('Unique words=', len(d)) 

"""""
Making Vocabulary and Training the Model
(sg=0 CBOW , sg=1 Skip-gram)

"""""
#########

model1 = Word2Vec(sent2, min_count=3,size= 300,workers=multiprocessing.cpu_count(), window =1, sg = 0)
print('Done Training')

SizeOfVocab = model1.wv.vocab
print('Size of Vocabulary=',len(SizeOfVocab))
print('Done making the Vocabulary')

#####
model2 = Word2Vec(sent2, min_count=3,size= 300,workers=multiprocessing.cpu_count(), window =1, sg = 1)
print('Done Training')

SizeOfVocab = model2.wv.vocab
print('Size of Vocabulary=',len(SizeOfVocab))
print('Done making the Vocabulary')

""""
Saving the embeddings and the model

"""""

from gensim.models import Word2Vec, KeyedVectors   
model1.wv.save_word2vec_format('W-CBOW-Stop.bin.gz', binary=True)
model1.wv.save_word2vec_format('W-CBOW-Stop.txt', binary=False)
model1.save('W-CBOW-Stop.bin')
print('Done Saving Model1')
#####
model2.wv.save_word2vec_format('W-Skip-Stop.bin.gz', binary=True)
model2.wv.save_word2vec_format('W-Skip-Stop.txt', binary=False)
model2.save('W-Skip-Stop.bin')
print('Done Saving Model2')

#model.save('model2.bin')

print('Done Saving the Embeddings')


