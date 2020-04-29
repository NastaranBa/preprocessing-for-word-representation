# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:05:04 2019

@author: nasba
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:16:34 2019

@author: nasba
"""

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset, Lemma
import re
import csv
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data

import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import csv
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset, Lemma
import re
import gc

News = pd.read_csv("C:/Users/nasba/Downloads/News/News.csv")

#num_reviews = openfile["review"].size
#num_reviews = str(News['review'][i])
# Initialize an empty list to hold the clean reviews
#clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
#for i in range( 0, len(num_reviews) ):
#    pos_khar = pos_tagger(nltk.word_tokenize(num_reviews))
#    clean_train_reviews.append(pos_khar)


#len(News['content'].unique().tolist())

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
#        review = re.sub(r'\W+', ' ', str(X[i])) #remove punc
        review = re.sub(r'[@%*~#+á\xc3\xa1\-\.\_\']','',str(X[i]))
        review = re.sub(r'\s+', ' ', review)
        review = re.sub(r'\d+',' ', review)# remove numbers
        review = review.lower() #lower case
        review = re.sub(r'\s+', ' ', review)
        corpus.append(review)        
#    return corpus        
    #Tokenizing and Word Count  
    words=[]
    for i in range(len(corpus)):
        words= nltk.word_tokenize(corpus[i])
        #sentences.append(words)
    gc.collect()
    return words

X = [[el] for el in X] 

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
sentences = Parallel(n_jobs=num_cores)(delayed(majid)(i) for i in X)
#############

class AntonymReplacer(object):
  def __init__(self):
    self.fname= 'C:/Users/nasba/Downloads/antonyms.csv'
    self.word_map = {}
    for line in csv.reader(open(self.fname)):
        word, ant = line
        word = word.strip()
        ant= ant.strip() 
        self.word_map[word] = ant
  
  def replace(self, word):
    pos=None
    antonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
      for lemma in syn.lemmas():
        for antonym in lemma.antonyms():
          antonyms.add(antonym.name())
    if len(antonyms) == 1:
      gc.collect()
      return antonyms.pop()
    else:
        #print(word)
        if word in self.word_map:
            #print(self.word_map[word])
            gc.collect()
            return self.word_map[word]
        gc.collect()
        return None
  


  def replace_negations(self, sent):
     if 'not' in sent or 'never' in sent or 'nor' in sent or 'neither' in sent:
        words = []
        i, l = 0, len(sent)
        while i < l:
          word = sent[i] #not
          if i+1 < l and (word == 'not' or word == 'never' or word == 'nor' or word == 'neither') :
            ant = self.replace(sent[i+1])
            if ant:
              words.append(ant)
              i += 2
              continue
          words.append(word)
          i += 1
        gc.collect()
        return words
     else:
        gc.collect()
        return sent

  

    
def majid2(X):
    replacer = AntonymReplacer()
    gc.collect()
    return replacer.replace_negations(X)
 
from joblib import Parallel, delayed
import multiprocessing


    
num_cores = multiprocessing.cpu_count()
sent2 = Parallel(n_jobs=num_cores)(delayed(majid2)(i) for i in sentences)

######################
#####################
count = 0
c = {}
for words in sent1:
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

model1 = Word2Vec(sent, min_count=3,size= 300,workers=multiprocessing.cpu_count(), window =1, sg = 0)
print('Done Training')

SizeOfVocab = model1.wv.vocab
print('Size of Vocabulary=',len(SizeOfVocab))
print('Done making the Vocabulary')

#####
model2 = Word2Vec(sent, min_count=3,size= 300,workers=multiprocessing.cpu_count(), window =1, sg = 1)
print('Done Training')

SizeOfVocab = model2.wv.vocab
print('Size of Vocabulary=',len(SizeOfVocab))
print('Done making the Vocabulary')

""""
Saving the embeddings and the model

"""""

from gensim.models import Word2Vec, KeyedVectors   
model1.wv.save_word2vec_format('W-CBOW-Neg-nop-nos.bin.gz', binary=True)
model1.wv.save_word2vec_format('W-CBOW-Neg-nop-nos.txt', binary=False)
model1.save('W-CBOW-Neg-nop-nos.bin')
print('Done Saving Model1')
#####
model2.wv.save_word2vec_format('W-Skip-Neg-nop-nos.gz', binary=True)
model2.wv.save_word2vec_format('W-Skip-Neg-nop-nos.txt', binary=False)
print('Done Saving Model2')

#model.save('model2.bin')

print('Done Saving the Embeddings')