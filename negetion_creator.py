"""
@author: Nastaran Babanejad
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


len(sentences)



#load dictionary
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
            
#load UKWac dictionary with word frequency
filepath = '.../input/UKWac.sorted.uk.word.unigrams.csv'  
word_freq = {}   
with open(filepath, encoding="utf8") as f:
    for line in f:
        line = line.rstrip()
        if line:
            x = line.split(',')
            print(x)
            #print(key, val)
            word_freq[x[0]] = str(x[1])



def bigram_corr(line):
    words = line.split() #split line into words and freq
    for idx, (word1, word2) in enumerate(zip(words[:-1], words[1:])):
        for i,j in word_map: #iterate over multiple choices
            if (word2==j) and ((word1,i) < (word1,j)): #if 2nd words of both match, and 1st word is at an edit distance of 2 or 1high freq, replace word with highest occurring freq
                break
    return " ".join(words)


def replace( word):
    pos=None
    antonyms = set()
    if word in word_map:
	    #print(word_map[word])
	    gc.collect()
	    return word_map[word]
    else:
	for syn in wordnet.synsets(word, pos=pos):
	    for lemma in syn.lemmas():
		for antonym in lemma.antonyms():
		    antonyms.add(antonym.name())

	if len(antonyms) == 1:
	    #print(antonyms)
	    gc.collect()
	    word_map[word] = antonyms.pop()
	    return word_map[word]
	else:
	    gc.collect()
	    return None


import re 
def replace_negations( sent, word_map):
    idx = 0
    indices = []
    indices = [i for i, x in enumerate(sent) if x == 'not' or x == 'never' or x == 'nor' or x == 'neither']
    #print(indices)
    c = 0
    for i in indices:
	l = len(sent)
	i -= c
	#print(word)
	if i >= 0 and i+1 < l : 
		if sent[i+1] in word_map and word_map[sent[i+1]] != 'None':
		    sent[i] = word_map[sent[i+1]]
		    sent.pop(i +1 )
		    c += 1




gc.collect()



len(sentences)



import time
import multiprocessing 
from multiprocessing import Process, freeze_support,  Manager


from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

def negation_multiprocessor(d,sent):
    for x in sent:
        replace_negations(x, d)


processes = []
for i,v in [(0,17000), (17000,34000), (34000,51000), (51000,68000), (68000,85000), (85000,102000), (102000,119000), (119000, 142546)]:
    sent = sentences[i:v]
    p = multiprocessing.Process(target=negation_multiprocessor, args=( word_map, sent))
    processes.append(p)
    p.start()


for process in processes:
    process.join()







