
import sys, math, re
from operator import itemgetter
import math
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import sklearn.datasets
import sklearn.metrics
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def clean_dataset(dataframe_dataset):

# The list for the corresponding 1 and 0 values of the comments
    dataframe_target = dataframe_dataset[['Insult']]
    insult_target = dataframe_target.ix[:,0].tolist()

# Removing \n
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('\n', '')

# Removing all HTML Tags
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('[^\w\s]','')

# Converting all uppercase letters to lower case
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.lower()

#Tokenizing all the words in a given sentence
    dataframe_dataset['tokenized_sents'] = dataframe_dataset.apply(lambda row: nltk.word_tokenize(row['Comment']), axis=1)

#Removing all the stop words which have no meaning
    stop = stopwords.words('english')
    dataframe_dataset['tokenized_sents'] = dataframe_dataset['tokenized_sents'].apply(lambda x: [item for item in x if item not in stop])

# Stemming all the words
    stemmer = SnowballStemmer('english')
    dataframe_dataset['stemmed'] = dataframe_dataset["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])

# Separating the list of both insulting and non insulting comments
    '''for i in dataframe_dataset['Insult']:
        if dataframe_dataset.iloc[i]['Insult'] == 1:

            insulting_list.append(dataframe_dataset['stemmed'])

    print(insulting_list)'''
    '''dataframe_dataset_filter = dataframe_dataset.loc[dataframe_dataset['Insult'] == 1]
    insulting_list = dataframe_dataset_filter['stemmed'].tolist()
    print(insulting_list)'''

# Replacing shorthand and other internet slangs with correct phrases
    map_words = {"u": "you", "em":"them", "da":"the", "yo":"you",
            "ur":"you", "won't": "will not", "won't": "will not",
            "can't": "can not", "i'm": "i am", "i'm": "i am", "ain't": "is not",
            "'ll": "will", "'t": "not", "'ve": "have", "'s": "is", "'re": "are",
            "'d": "would"}

    dataframe_dataset['replaced stemmed'] = dataframe_dataset["stemmed"].map(lambda x: [map_words[x[i]] if x[i] in map_words else x[i] for i in range(len(x)) ])

    return




inp = open(path, 'r')
inp_header = inp.readline()
header_list = re.split("[,\n]" , inp_header)


# Taking google list of bad words
bw = open('full-list-of-bad-words.txt', 'r')

#inp_text = bw.read()  # raeding file generated from puzzleenerator.py
#inp_text = re.split('\n|,', inp_text)
#print(inp_text)

#print dataframe_dataset.applymap(lambda x: isinstance(x, (int, float))).all(0)

#Reading Train Dataset
path = r'./train.csv'
print("Path to dataset is \n" + path)
dataframe_dataset = pd.read_csv(path, na_values='unknown', encoding="utf-8")
clean_dataset(dataframe_dataset)
dataframe_dataset.to_csv('Train_clean.csv')

#Reading Test Dataset
path = r'./Test/test_with_solutions.csv'
print("Path to test dataset is \n" + path)
dataframe_dataset = pd.read_csv(path, na_values='unknown', encoding="utf-8")
clean_dataset(dataframe_dataset)
dataframe_dataset.to_csv('Train_clean.csv')

