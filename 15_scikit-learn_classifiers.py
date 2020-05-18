# Importing the libraries and modules that will be needed
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier # scikit-learn module in NLTK
# machine learning algorithms from scikit-learn that will be applied here
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


# This is the Python list that will be populated with all the reviews
reviews = []

# In movie_reviews.categories() we have 2 categories: 'neg' and 'pos'
for category in movie_reviews.categories():
    # file ID examples: 'neg/cv985_5964.txt' or 'pos/cv533_9821.txt'
    # 1,000 negative ('neg') and 1,000 positive ('pos') reviews
    for fileid in movie_reviews.fileids(category):
        # Appending with tuple (list_of_words_per_review, category)
        reviews.append((list(movie_reviews.words(fileid)), category))

# This is the Python list that will be populated with all the words in all the reviews
all_words = []

# Creating a list of stop words so those are excluded from our all_words list
# The parameter for stopwords.words() needs to be the language we use
stop_words = stopwords.words("english")

# Adding some more custom stop words
more_stop_words = ['.', ',', ':', ';', '?', '!', '-', '\"', '\'', '(', ')']
stop_words += more_stop_words # Adding our custom list to the built-in list

# Shuffling the reviews for training purposes
random.shuffle(reviews)

# Looping through all the words from all the reviews
for word in movie_reviews.words():
    # Excluding the stop words as we did in 02_stop_words.py
    if word not in stop_words:
        all_words.append(word.lower()) # better not to distinguish lower and upper

# Now, we can convert our list of words into an NLTK frequency distribution list
all_words = nltk.FreqDist(all_words)

# Retrieving the most frequent 3,000 elements into 'word_features'
word_features = list(all_words.keys())[:3000]

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
# Here we will be using the SklearnClassifier module and the scikit-learn library

# Creating a Python dictionary with classifiers to loop through them
# The key is our internal name, and the value is the class name from scikit-learn
classifiers = {'MultinomialNB_classifier': MultinomialNB,
 'BernoulliNB_classifier': BernoulliNB,
 'LogisticRegression_classifier': LogisticRegression,
 'SGDClassifier_classifier': SGDClassifier,
 'SVC_classifier': SVC,
 'LinearSVC_classifier': LinearSVC,
 'NuSVC_classifier': NuSVC,
 }

# Looping through our Python dictionary
for name, classifier in classifiers.items():
    # Retrieve the algorithm name because it will be converted into an object of the SklearnClassifier class
    algorithm_name = name
    name = SklearnClassifier(classifier()) # creating the object of the classifier class
    name.train(training_set) # train against our training set
    print('%s accuracy (percentage):' % (algorithm_name), nltk.classify.accuracy(name, testing_set) * 100) # try it out with the testing set
