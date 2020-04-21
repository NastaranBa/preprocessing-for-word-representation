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
import gc
'''
POS-TAGGER, returns NAVA words
'''

News = pd.read_csv("C:.../input/News.csv")

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
        review = re.sub(r'\d+','', str(X[i]))# remove number
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

def pos_tagger(sentences):
    tags = [] #have the pos tag included
    nava_sen = []
    pt = nltk.pos_tag(sentences)
#     for s in sentences:
#     s_token = nltk.word_tokenize(sentences)
#     pt = nltk.pos_tag(s_token)
    nava = []
    nava_words = []
    for t in pt:
        if t[1].startswith('NN') or t[1].startswith('NNS') or t[1].startswith('NNP') or t[1].startswith('NNPS') or t[1].startswith('JJ') or t[1].startswith('JJR') or t[1].startswith('JJS') or  t[1].startswith('VB') or t[1].startswith('VBG') or t[1].startswith('VBN') or t[1].startswith('VBP') or t[1].startswith('VBZ') or t[1].startswith('RB') or t[1].startswith('RBR') or t[1].startswith('RBS'):
            nava.append(t)
            nava_words.append(t[0])
    return nava_words

def majid2(X):
    review = pos_tagger(X)
    gc.collect()
    return review
 
from joblib import Parallel, delayed
import multiprocessing


    
num_cores = multiprocessing.cpu_count()
sent2 = Parallel(n_jobs=num_cores)(delayed(majid2)(i) for i in sentences)

    
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
model1.wv.save_word2vec_format('W-CBOW-POS.bin.gz', binary=True)
model1.wv.save_word2vec_format('W-CBOW-POS.txt', binary=False)
model1.save('W-CBOW-POS.bin')
print('Done Saving Model1')
#####
model2.wv.save_word2vec_format('W-Skip-POS.gz', binary=True)
model2.wv.save_word2vec_format('W-Skip-POS.txt', binary=False)
model1.save('W-Skip-POS.bin')
print('Done Saving Model2')

#model.save('model2.bin')

print('Done Saving the Embeddings')

