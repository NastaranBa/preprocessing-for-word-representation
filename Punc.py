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


"""
Importing Dataset

"""

News = pd.read_csv("C:.../input/News.csv")


print('Done Importing')



#####################
##Punc
####################

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

News['content'] = News['content'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
##########################################

text= News['content']
X=text.values.tolist()

########################
def majid(X):
    corpus = []
    for i in range(0, len(X)):
        #Normally removing punctuation
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(X[i])) #remove punctuation
        review = re.sub(r'\d+','', str(X[i]))# remove number
        #review = review.lower() #lower case
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

count = 0
c = {}
for words in sentences:
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

model1 = Word2Vec(sentences, min_count=5,size= 300,workers=multiprocessing.cpu_count(), window =1, sg = 0)
print('Done Training')

SizeOfVocab = model1.wv.vocab
print('Size of Vocabulary=',len(SizeOfVocab))
print('Done making the Vocabulary')

#####
model2 = Word2Vec(sentences, min_count=5,size= 300,workers=multiprocessing.cpu_count(), window =1, sg = 1)
print('Done Training')

SizeOfVocab = model2.wv.vocab
print('Size of Vocabulary=',len(SizeOfVocab))
print('Done making the Vocabulary')

""""
Saving the embeddings and the model

"""""

from gensim.models import Word2Vec, KeyedVectors   
model1.wv.save_word2vec_format('W-CBOW-Punc.bin.gz', binary=True)
model1.wv.save_word2vec_format('W-CBOW-Punc.txt', binary=False)
model1.save('W-CBOW-Punc.bin')
print('Done Saving Model1')
#####
model2.wv.save_word2vec_format('W-Skip-Punc.gz', binary=True)
model2.wv.save_word2vec_format('W-Skip-Punc.txt', binary=False)
model2.save('W-Skip-Punc.bin')
print('Done Saving Model2')

#model.save('model2.bin')

print('Done Saving the Embeddings')
