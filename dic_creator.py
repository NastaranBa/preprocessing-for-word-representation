# -*- coding: utf-8 -*-
"""

@author: nasba
"""

import nltk
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
nltk.download('punkt')
nltk.download('omw')
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset, Lemma
import re
import gc

News = pd.read_csv(".../input/News.csv")


print('Done Importing')

text= News['content']
X=text.values.tolist()

def majid(X):
    corpus = []
    for i in range(0, len(X)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(X[i])) #remove punctuation
        review = re.sub(r'\d+',' ', review)# remove number
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

import time
import multiprocessing 
from multiprocessing import Process, freeze_support


fname= '.../input/antonyms.csv'


word_map = {}

for line in csv.reader(open(fname)):
    word, ant = line
    word = word.strip()
    ant= ant.strip() 
    word_map[word] = ant

def create_dic(sent, word_map):
    pos=None
    antonyms = set()
    i, l = 0, len(sent)
    indices = [i for i, x in enumerate(sent) if x == 'not' or x == 'never' or x == 'nor' or x == 'neither']
    for i in indices:
if i+1 < l:
    word = sent[i+1]
    if word not in word_map:
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())

        if len(antonyms) == 1:
            #print(antonyms)
            gc.collect()
            word_map[word] = antonyms.pop()
        else:
            word_map[word] = 'None'

gc.collect()

import time
import multiprocessing 
from multiprocessing import Process, freeze_support,  Manager


def create_dic_multiprocessor(d,sent):
    i = 0
    #word_map = {}
    #print(sent)
    for x in sent:
        create_dic(x, d)
        i += 1
        if (i % 1000) == 0:
            print(i)

processes = []
manager = Manager()
d = manager.dict()
for i,v in [(0,17000), (17000,34000), (34000,51000), (51000,68000), (68000,85000), (85000,102000), (102000,119000), (119000, 142546)]:
    sent = sentences[i:v]
    p = multiprocessing.Process(target=create_dic_multiprocessor, args=( d, sent))
    processes.append(p)
    p.start()


for process in processes:
    process.join()


#copy to word_map dic
for k,v in d.items():
    word_map[k] = v


#write to dic
w = csv.writer(open(".../input/dic.csv", "w",newline='', encoding = 'utf8'))
for key, val in word_map.items():
    #print(key, val)
    w.writerow([key, val])

#read from dic
filepath = '.../input/dic.csv'  
word_map = {}   
with open(filepath, encoding="utf8") as f:
    for line in f:
        line = line.rstrip()
        if line:
            x = line.split(',')
            print(x)
            #print(key, val)
            word_map[x[0]] = str(x[1])



len(word_map)





