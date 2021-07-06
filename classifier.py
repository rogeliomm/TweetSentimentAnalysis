import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize

import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

## Long Reviews
# documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         documents.append(list(movie_reviews.words(fileid)), category)

# random.shuffle(documents)

# all_words = []
# for w in movie_reviews.words():
#     all_words.append(w.lower())

## Short reviews
short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()

# documents = []

# for r in short_pos.split('\n'):
#     documents.append( (r, "pos") )

# for r in short_neg.split('\n'):
#     documents.append( (r, "neg") )

# all_words = []
# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)

# for w in short_pos_words:
#     all_words.append(w.lower())

# for w in short_neg_words:
#     all_words.append(w.lower())

all_words = []
documents = []

# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickled/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

save_word_features = open("pickled/word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featureset = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featureset)

training_set = featureset[:10000]
testing_set = featureset[10000:]

save_featureset = open("pickled/featureset.pickle", "wb")
pickle.dump(featureset, save_featureset)
save_featureset.close()

# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

## Original Naive Bayes
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Original Naive Bayes accuracy percent: ", (nltk.classify.accuracy(classifier,testing_set)) * 100)
# classifier.show_most_informative_features(15)

# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

## Mulinomial naive bayes
MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training_set)
print("MultinomialNB accuracy percent: ", (nltk.classify.accuracy(MultinomialNB_classifier,testing_set)) * 100)

save_classifier = open("pickled/MultinomialNB.pickle", "wb")
pickle.dump(MultinomialNB_classifier, save_classifier)
save_classifier.close()

## Bernouli naive bayes
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set)) * 100)

save_classifier = open("pickled/BernoulliNB.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

## LogisticRegression
# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set)) * 100)

# save_classifier = open("LogisticRegression.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

## SGDClassifier
# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set)) * 100)

## SVC
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC accuracy percent: ", (nltk.classify.accuracy(SVC_classifier,testing_set)) * 100)

## LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set)) * 100)

save_classifier = open("pickled/LinearSVC.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

## NuSVC
# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set)) * 100)

# save_classifier = open("NuSVC.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

voted_classifier = VoteClassifier(MultinomialNB_classifier, BernoulliNB_classifier, LinearSVC_classifier)
print("Voted classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier,testing_set)) * 100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), " Confidence percent: ", voted_classifier.confidence(testing_set[0][0]) * 100)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats)