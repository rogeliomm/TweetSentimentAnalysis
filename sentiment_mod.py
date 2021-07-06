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

## Short reviews
short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()

## Documents
documents_f = open("pickled/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

## Word Features
word_features5k_f = open("pickled/word_features.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

## Featureset
featuresets_f = open("pickled/featureset.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

## Models

## Mulinomial naive bayes
open_file = open("pickled/MultinomialNB.pickle", "rb")
MultinomialNB_classifier = pickle.load(open_file)
open_file.close()

## Bernouli naive bayes
open_file = open("pickled/BernoulliNB.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

## LinearSVC
open_file = open("pickled/LinearSVC.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(MultinomialNB_classifier, BernoulliNB_classifier, LinearSVC_classifier)
print("Voted classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier,testing_set)) * 100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), " Confidence percent: ", voted_classifier.confidence(testing_set[0][0]) * 100)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)