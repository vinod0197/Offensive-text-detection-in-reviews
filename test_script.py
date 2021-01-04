# Using sklearn features to train the data
import pandas as pd
import sys
import string
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.datasets
import sklearn.metrics
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import os
import nltk
import re
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import load_files
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import _pickle as cPickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.base import TransformerMixin
import math
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

bw_list = []
invalid_input = True

class BadWordTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        X = X.astype('str').map(lambda x: x.split() if (x!='nan') else '')

        return pd.DataFrame(X.map(lambda x: len(list(filter((lambda y: y in bw_list), x) ))))


    def fit(self, X, y=None, **fit_params):
        return self

with open(r"my_classifier.pkl", "rb") as input_file:
    text_clf = cPickle.load(input_file)

bw = open('bad_words.txt', 'r')
inp_text = bw.read()  # reading file generated from puzzleenerator.py
inp_text = re.split('\n|,', inp_text)
bw_list= [word.replace( u"\xa0",u"") for word in inp_text]
target_names = ['Not an insult', 'Insult']
'''
# Training
dataframe_dataset = pd.read_csv("./Train_clean.csv", na_values='unknown', encoding="utf-8")
bw = open('bad_words.txt', 'r')
inp_text = bw.read()  # raeding file generated from puzzleenerator.py
inp_text = re.split('\n|,', inp_text)

bw_list= [word.replace( u"\xa0",u"") for word in inp_text]


dataframe_dataset["joined replaced stemmed"].fillna(" ", inplace=True)
dataframe_dataset["bad words"] = dataframe_dataset["joined replaced stemmed"]. \
    astype('str').map(lambda x: x.split() if (x!='nan') else '')

dataframe_dataset["bad words"] = dataframe_dataset["bad words"]. \
    map(lambda x: len(list(filter((lambda y: y in bw_list), x) )))

print("Using Naive Baye's")
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
            ('counts', CountVectorizer(ngram_range=(2, 2))),
            ('tf_idf', TfidfTransformer())
            ])),
        ('badwords', BadWordTransformer())
        ])),
    ('classifier', MultinomialNB())
    ])

text_clf = text_clf.fit(dataframe_dataset["joined replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value")
print("Print metric report")
'''

# Testing
dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")
dataframe_dataset_test["joined replaced stemmed"].fillna(" ", inplace=True)

dataframe_dataset_test["bad words"] = dataframe_dataset_test["joined replaced stemmed"]. \
    astype('str').map(lambda x: x.split() if (x!='nan') else '')

dataframe_dataset_test["bad words"] = dataframe_dataset_test["bad words"]. \
    map(lambda x: len(list(filter((lambda y: y in bw_list), x) )))
print(type(dataframe_dataset_test["joined replaced stemmed"]))
predicted = text_clf.predict(dataframe_dataset_test["joined replaced stemmed"])
print(np.mean(predicted == dataframe_dataset_test["Insult"]))
print(metrics.classification_report(dataframe_dataset_test["Insult"], predicted, target_names=target_names))


print ("\n Welcome to the user test program for insult. Fear me not. Won't insult you back. I promise ;)\n")
while invalid_input:
    op = input ("Please input your comment : ")
    if op == None:
        invalid_input = False
    op_series = pd.Series(op)

    predicted_new = text_clf.predict(op_series)

    print("Insult") if predicted_new == 1 else print("Not insult")

    print("Do you wish to continue?")
    yorno = input ("y/N")
    if yorno == 'y' :
        invalid_input = True
    else :
        invalid_input = False

