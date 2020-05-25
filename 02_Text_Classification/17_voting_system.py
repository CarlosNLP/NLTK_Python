# Importing the libraries and modules that will be needed
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI # we will create our own class inheriting from this one

import random

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from statistics import mode # we will choose which result has the most votes among the classifiers

# Creating our own class based (inheriting) on ClassifierI
class VoteClassifier(ClassifierI):
    def __init__(self, classifiers): # we will pass a list of classifiers through VoteClassifier
        self._classifiers = classifiers

    def classify(self, features): # we pass the features we created previously
        votes = []
        for c in self._classifiers: # running through each classifier
            vote = c.classify(features) # 'pos' or 'neg'
            votes.append(vote) # populating the "votes" list with all the votes
        return mode(votes) # whatever the majority is: 'pos' or 'neg'

    def get_confidence(self, features): # get the confidence of our choice (how many classifiers out of the total said "pos" or "neg")
        votes = []
        for c in self._classifiers: # running through each classifier
            vote = c.classify(features) # 'pos' or 'neg'
            votes.append(vote) # populating the "votes" list with all the votes
        choice_votes = votes.count(mode(votes)) # number of votes of the chosen one (e.g.: how many times 'pos' or 'neg' was chosen)
        confidence = choice_votes / len(votes) # getting the confidence by dividing the choice_votes by  the total number of votes (e.g.: 5/7)
        return confidence

# This is the Python list that will be populated with all the reviews
reviews = []

print('Loading our reviews...')
# In movie_reviews.categories() we have 2 categories: 'neg' and 'pos'
for category in movie_reviews.categories():
    # file ID examples: 'neg/cv985_5964.txt' or 'pos/cv533_9821.txt'
    # 1,000 negative ('neg') and 1,000 positive ('pos') reviews
    for fileid in movie_reviews.fileids(category):
        # Appending with tuple (list_of_words_per_review, category)
        reviews.append((list(movie_reviews.words(fileid)), category))

# Shuffling the reviews for training purposes
random.shuffle(reviews)

# Creating a list of stop words so those are excluded from our all_words list
# The parameter for stopwords.words() needs to be the language we use
stop_words = stopwords.words("english")

# Adding some more custom stop words
more_stop_words = ['.', ',', ':', ';', '?', '!', '-', '\"', '\'', '(', ')']
stop_words += more_stop_words # Adding our custom list to the built-in list

# This is the Python list that will be populated with all the words in all the reviews
all_words = []

# Looping through all the words from all the reviews
print('Appending our words excluding the stop words...')
for word in movie_reviews.words():
    # Excluding the stop words as we did in 02_stop_words.py
    if word not in stop_words:
        all_words.append(word.lower()) # better not to distinguish lower and upper

# Having our full list of words, we will get their part-of-speech tagging
print('Tagging our full list of words...')
tagging = nltk.pos_tag(all_words)

# To improve the model, we will get rid of some words depending on their POS tag
# We recreate all_words only with adjectives
all_words = []

print('Retrieving only the adjectives and dumping them into our all_words list...')
for i in range(len(tagging)):
    if tagging[i][1].startswith("J"): # taking adjectives
        all_words.append(tagging[i][0])
##    elif tagging[i][1].startswith("RB"): # taking adverbs
##        all_words.append(tagging[i][0])
##    elif tagging[i][1].startswith("N"): # taking nouns
##        all_words.append(tagging[i][0])

# Now, we can convert our list of words into an NLTK frequency distribution list
all_words = nltk.FreqDist(all_words)
print('Len of all_words:', len(all_words))
print(all_words.most_common(15))

# Retrieving the 3,000 most common elements into 'word_features'
word_features = list(all_words.keys())[:3000]

# Method to convert from words to features
def find_features(review):
    words = set(review) # it converts the list of words (with duplicates) into unique words (no duplicates)
    features = {} # this will be our Python dictionary with the features
    for w in word_features: # looping through the 3,000 most common words
        # The key of the Python dictionary will be the word itself
        # The value of the Python dictionary will be True or False, depending on whether the word itself can be found in the specific review
        features[w] = (w in words)
    return features # returning the 'features' dictionary

# Preparing data for the training
# We will convert the original list 'reviews' with tuples (list_of_words, category)
# into a new list 'feature_sets' with tuples (features_dictionary_for_specific_review, category)
print('Getting our model features...')
feature_sets = []
for review, category in reviews:
    feature_sets.append((find_features(review), category))

# Having the feature_sets list ready, let's create the training and testing sets
training_set = feature_sets[:1900] # 1,900 reviews previously shuffled
testing_set = feature_sets[1900:] # 100 reviews previously shuffled

# Creating the Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# Checking accuracy of the model against the testing set
print('Naive Bayes accuracy (percentage):', nltk.classify.accuracy(classifier, testing_set) * 100)

# Showing the most 15 informative features
classifier.show_most_informative_features(15)

# Next objective: try different algorithms
# Here we will be using the SklearnClassifier module from NLTK and the scikit-learn library

# Creating a Python dictionary with classifiers to loop through them
# The key is our internal name, and the value is the class name from scikit-learn, imported in the first lines
classifiers = {'MultinomialNB_classifier': MultinomialNB,
 'BernoulliNB_classifier': BernoulliNB,
 'LogisticRegression_classifier': LogisticRegression,
 'SGDClassifier_classifier': SGDClassifier,
 'SVC_classifier': SVC,
 'LinearSVC_classifier': LinearSVC,
 'NuSVC_classifier': NuSVC,
 }

# List of trained classifiers
trained_classifiers = []

# Looping through our Python dictionary with classifiers
for name, classifier in classifiers.items():
    # Retrieve the algorithm name because it will be converted into an object of the SklearnClassifier class
    algorithm_name = name
    name = SklearnClassifier(classifier()) # creating the object of the classifier class
    name.train(training_set) # train against our training set
    trained_classifiers.append(name)
    print('%s accuracy (percentage):' % (algorithm_name), nltk.classify.accuracy(name, testing_set) * 100) # try it out with the testing set

# Creating our object from the VoteClassifier class created by us
voted_classifier = VoteClassifier([c for c in trained_classifiers])
print('voted_classifier accuracy %:', nltk.classify.accuracy(voted_classifier, testing_set) * 100) # try it out with the testing set

# Trying out some of the examples from the testing set
print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.get_confidence(testing_set[0][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:", voted_classifier.get_confidence(testing_set[1][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:", voted_classifier.get_confidence(testing_set[2][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:", voted_classifier.get_confidence(testing_set[3][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:", voted_classifier.get_confidence(testing_set[4][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:", voted_classifier.get_confidence(testing_set[5][0]) * 100)



