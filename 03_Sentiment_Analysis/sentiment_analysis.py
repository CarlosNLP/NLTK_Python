# Importing the libraries and modules that will be needed
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier # scikit-learn module in NLTK
from nltk.classify import ClassifierI # we will create our own class inheriting from this one

import random
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from statistics import mode # we will choose which result has the most votes among the classifiers

# Creating our own class based (inheriting) on ClassifierI
class VoteClassifier(ClassifierI):
    def __init__(self, classifiers): # we will pass a list of classifiers through VoteClassifier
        self._classifiers = classifiers

    def classify(self, features): # we pass the features we created previously
        votes = []
        for c in self._classifiers: # running through each classifier
            vote = c.classify(features) # 'pos' or 'neg'
            votes.append(vote)
        return mode(votes) # who got the most votes

    def get_confidence(self, features): # get the confidence of our choice (how many classifiers out of the total said "pos" or "neg")
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)
        choice_votes = votes.count(mode(votes)) # number of votes of the chosen one
        confidence = choice_votes / len(votes) # (example: 5/7)
        return confidence

# Loading pickled reviews
reviews_f = open("pickled/reviews.pickle", "rb")
reviews = pickle.load(reviews_f)
reviews_f.close()

# Loading pickled word features
word_features_f = open("pickled/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()
print('# of word features:', len(word_features))

# Method to convert from words to features
def find_features(review):
    words = word_tokenize(review)
    features = {} # this will be our Python dictionary with the features
    for w in word_features: # looping through the 5,000 most common words
        # The key of the Python dictionary will be the word itself
        # The value of the Python dictionary will be True or False, depending on whether the word itself can be found in the specific review
        features[w] = (w in words)
    return features # returning the 'features' dictionary

# Loading pickled feature sets
feature_sets_f = open("pickled/feature_sets.pickle", "rb")
feature_sets = pickle.load(feature_sets_f)
feature_sets_f.close()
print('# of feature sets:', len(feature_sets))

# Shuffling feature sets
random.shuffle(feature_sets)

# Having the feature_sets list ready, let's create the training and testing sets
training_set = feature_sets[:10000] # 10,000 reviews previously shuffled
testing_set = feature_sets[10000:] # Around 600 reviews previously shuffled

trained_classifiers = []

# Loading all the classifiers
open_file = open("pickled/classifier.pickle", "rb")
classifier = pickle.load(open_file)
# trained_classifiers.append(classifier) # we do not include the default classifier because we want the list to be an odd number (to get the most voted)
open_file.close()

open_file = open("pickled/MultinomialNB_classifier5K.pickle", "rb")
MultinomialNB_classifier = pickle.load(open_file)
trained_classifiers.append(MultinomialNB_classifier)
open_file.close()

open_file = open("pickled/BernoulliNB_classifier5K.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
trained_classifiers.append(BernoulliNB_classifier)
open_file.close()

open_file = open("pickled/LogisticRegression_classifier5K.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
trained_classifiers.append(LogisticRegression_classifier)
open_file.close()

open_file = open("pickled/SGDClassifier_classifier5K.pickle", "rb")
SGDClassifier_classifier = pickle.load(open_file)
trained_classifiers.append(SGDClassifier_classifier)
open_file.close()

open_file = open("pickled/LinearSVC_classifier5K.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
trained_classifiers.append(LinearSVC_classifier)
open_file.close()

# Creating our object from the VoteClassifier class created by us
voted_classifier = VoteClassifier([c for c in trained_classifiers])

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.get_confidence(feats)

