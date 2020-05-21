# Importing the libraries and modules that will be needed
import nltk
import random # module that will help us shuffling the reviews
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

# This is the Python list that will be populated with all the reviews
reviews = []

# In movie_reviews.categories() we have 2 categories: 'neg' and 'pos'
for category in movie_reviews.categories():
    # file ID examples: 'neg/cv985_5964.txt' or 'pos/cv533_9821.txt'
    # 1,000 negative ('neg') and 1,000 positive ('pos') reviews
    for fileid in movie_reviews.fileids(category):
        # Appending with tuple (list_of_words_per_review, category)
        reviews.append((list(movie_reviews.words(fileid)), category))

# Next objective: create a frequency distribution list to get the most common words
# This is the Python list that will be populated with all the words in all the reviews
all_words = []

# Creating a list of stop words so those are excluded from our all_words list
# The parameter for stopwords.words() needs to be the language we use
stop_words = stopwords.words("english")

# Adding some more custom stop words
more_stop_words = ['.', ',', ':', ';', '?', '!', '-', '\"', '\'', '(', ')']
stop_words += more_stop_words # Adding our custom list to the built-in list

# Since our 'reviews' list contains all the reviews in the same order, which is
# 1,000 negative and then 1,000 positive reviews, we will just shuffle them
# This will be useful if we want to train with these examples, or if we just
# need to pick a random shuffled review
random.shuffle(reviews)

# Looping through all the words from all the reviews
for word in movie_reviews.words():
    # Excluding the stop words as we did in 02_stop_words.py
    if word not in stop_words:
        all_words.append(word.lower()) # better not to distinguish lower and upper

# Printing the number of words. Note: they are not unique, there will be duplicates
print('# of words in movie_reviews.words():', len(all_words))

# Now, we can convert our list of words into an NLTK frequency distribution list
all_words = nltk.FreqDist(all_words)

# Printing the 10 most common words
# This provides a list of tuples (word, frequency)
print('\nMost common words:', all_words.most_common(10))

# Getting the number of occurrences of a specific word provided by the user
print('\n# of fun:', all_words["fun"])

# Retrieving the first 3,000 elements (words, since we are using .keys())
# all_words goes from most common to least common, so we get the most frequent 3,000 words here
# We could do it with all the words, but that would take much longer and words with much less frequency are not so useful
word_features = list(all_words.keys())[:3000]

# Method to convert from just words to features
def find_features(review):
    words = set(review) # it converts the list of words (with duplicates) into unique words (no duplicates)
    features = {} # this will be our Python dictionary with the features
    for w in word_features: # looping through the 3,000 most common words
        # The key of the Python dictionary will be the word itself
        # The value of the Python dictionary will be True or False, depending on whether the word itself can be found in the specific review
        features[w] = (w in words)
    return features # returning the 'features' dictionary

# Printing an example - this will give back the features dictionary applied to that specific review
print(find_features(movie_reviews.words('pos/cv533_9821.txt')))

# Preparing data for the training
# We will convert the original list 'reviews' with tuples (list_of_words, category)
# into a new list 'feature_sets' with tuples (features_dictionary_for_specific_review, category)
feature_sets = []
for review, category in reviews:
    feature_sets.append((find_features(review), category))

# Printing an example - this will give back a tuple with the following structure:
# (features_dictionary_for_specific_review, category)
print(feature_sets[166])
