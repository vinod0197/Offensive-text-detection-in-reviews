# Using sklearn features to train the data
# Training
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics



dataframe_dataset = pd.read_csv("./Train_clean.csv", na_values='unknown', encoding="utf-8")

print("Using Naive Baye's")
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
print(len(dataframe_dataset["replaced stemmed"]))
text_clf = text_clf.fit(dataframe_dataset["replaced stemmed"])

print("Print here the mean value")
print("Print metric report")
docs_test = twenty_test.data



predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

# save the classifier
with open('my_classifier.pkl', 'wb') as fid:
        cPickle.dump(text_clf, fid)

SVC_values = []


for n in range(100, train_data_length, 200):
    text_clf = text_clf.fit(twenty_train.data[:n], twenty_train.target[:n])
    predicted = text_clf.predict(docs_test)
    SVC_values.append(metrics.f1_score(twenty_test.target, predicted, average='weighted'))

