# Effects of Preprocessing in Word Embedding
A Comprehensive Analysis of Preprocessing for Word Representation Learning in Affective Tasks

1) We conduct a comprehensive analysis of the role of preprocessing techniques in affective tasks (including sentiment analysis, emotion classification and sarcasm detection), employing different models, over nine datasets.
2) We perform a comparative analysis of the accuracy performance of word vector models when preprocessing is applied at the training phase (training data) and/or at the downstream task phase (classification dataset).
3) We evaluate the performance of our best preprocessed word vector model against state-ofthe-art pretrained word embedding models.

List of Preprocessing Factors:
1) Punctuation (Punc)
2) Spelling correction (Spell)
3) Stemming (Stem)
4) Stopwords removal (Stop)
5) Negation (Neg)
6) Pos-Tagging (POS)

Training Models:
1) Word2Vec Skip-gram 
2) Word2Vec CBOW
3) BERT


Datasets used for Training:
1) News : https://www.kaggle.com/snapcrack/all-the-news
2) Wikipedia : https://www.kaggle.com/jkkphys/english-wikipedia-articles-20170820-sqlite

Datasets used for Classification:

a) Sentiment Analysis:
   1) IMDB : http://ai.stanford.edu/ amaas/data/sentiment/
   2) Semeval 2016 : http://alt.qcri.org/semeval2016/task4/index.php
   3) Airlines : https://www.kaggle.com/crowdflower/twitter-airlinesentiment
   
b) Emotion Detection:
   1) SSEC: http://www.romanklinger.de/ssec/
   2) ISEAR : https://github.com/PoorvaRane/Emotion-Detector/blob/master/ISEAR.csv
   3) Alm : http://people.rc.rit.edu/~coagla/affectdata/index.html
   
c) Sarcasm Detection:
   1) Onion : https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection
   2) IAC : https://nlds.soe.ucsc.edu/sarcasm2
   3) Reddit : https://nlp.cs.princeton.edu/SARC/0.0/
   
