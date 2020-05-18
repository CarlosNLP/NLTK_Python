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

# Having our full list of words, we will get their part-of-speech tagging
tagging = nltk.pos_tag(all_words)

# To improve the model, we will get rid of some words depending on their POS tag
# We recreate all_words only with adjectives
all_words = []

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

# Lists where we will store the scores for each algorithm
MultinomialNB_classifier_scores = []
BernoulliNB_classifier_scores = []
LogisticRegression_classifier_scores = []
SGDClassifier_scores = []
SVC_classifier_scores = []
LinearSVC_classifier_scores = []
NuSVC_classifier_scores = []

# Dictionary where we will store the scores (the above lists) for each algorithm, we will get an average afterwards
classifiers_score = {'MultinomialNB_classifier': MultinomialNB_classifier_scores,
 'BernoulliNB_classifier': BernoulliNB_classifier_scores,
 'LogisticRegression_classifier': LogisticRegression_classifier_scores,
 'SGDClassifier_classifier': SGDClassifier_scores,
 'SVC_classifier': SVC_classifier_scores,
 'LinearSVC_classifier': LinearSVC_classifier_scores,
 'NuSVC_classifier': NuSVC_classifier_scores,
 }

# We will be looping through the same 5 times to then get an average on the results
for i in range(5):
    print('\nROUND', i + 1, '...\n')
    # Shuffling feature sets - this will shuffle the review order again
    random.shuffle(feature_sets)

    # Assigning training and testing sets with suffled data
    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

    # Looping through our Python dictionary with classifiers
    for name, classifier in classifiers.items():
        # Retrieve the algorithm name because it will be converted into an object of the SklearnClassifier class
        algorithm_name = name
        name = SklearnClassifier(classifier()) # creating the object of the classifier class
        name.train(training_set) # train against our training set
        # Populating the corresponding classifier list with each score
        if algorithm_name == "MultinomialNB_classifier":
            MultinomialNB_classifier_scores.append(nltk.classify.accuracy(name, testing_set) * 100)
        elif algorithm_name == "BernoulliNB_classifier":
            BernoulliNB_classifier_scores.append(nltk.classify.accuracy(name, testing_set) * 100)
        elif algorithm_name == "LogisticRegression_classifier":
            LogisticRegression_classifier_scores.append(nltk.classify.accuracy(name, testing_set) * 100)
        elif algorithm_name == "SGDClassifier_classifier":
            SGDClassifier_scores.append(nltk.classify.accuracy(name, testing_set) * 100)
        elif algorithm_name == "SVC_classifier":
            SVC_classifier_scores.append(nltk.classify.accuracy(name, testing_set) * 100)
        elif algorithm_name == "LinearSVC_classifier":
            LinearSVC_classifier_scores.append(nltk.classify.accuracy(name, testing_set) * 100)
        elif algorithm_name == "NuSVC_classifier":
            NuSVC_classifier_scores.append(nltk.classify.accuracy(name, testing_set) * 100)
        print('%s accuracy (percentage):' % (algorithm_name), nltk.classify.accuracy(name, testing_set) * 100) # try it out with the testing set

# Introducing their scores into the Python dictionary
classifiers_score["MultinomialNB_classifier"] = list(MultinomialNB_classifier_scores)
classifiers_score["BernoulliNB_classifier"] = list(BernoulliNB_classifier_scores)
classifiers_score["LogisticRegression_classifier"] = list(LogisticRegression_classifier_scores)
classifiers_score["SGDClassifier_classifier"] = list(SGDClassifier_scores)
classifiers_score["SVC_classifier"] = list(SVC_classifier_scores)
classifiers_score["LinearSVC_classifier"] = list(LinearSVC_classifier_scores)
classifiers_score["NuSVC_classifier"] = list(NuSVC_classifier_scores)

# Method that will take a list of scores as a parameter, and will return the average
def get_average(list_of_scores):
    length = len(list_of_scores)
    total_sum = 0
    for score in list_of_scores:
        total_sum += score
    average = total_sum / length
    return average

print('\nPrinting averaged results...\n')

# Calling the get_average method to get the averaged results of each classifier
for name, scores in classifiers_score.items():
    print(name, get_average(scores))
    
