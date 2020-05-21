# Importing the libraries and modules that will be needed
import nltk
from nltk.corpus import movie_reviews # module with 2,000 movie reviews
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

# Printing the 150th example for reference
# The result is a tuple ([word1, word2, word3, ...], 'pos'), for example
print('Printing the tuple (list_of_words_per_review, category) in 150th example:', reviews[150])

# Next objective: create a frequency distribution list to get the most common words
# This is the Python list that will be populated with all the words
all_words = []

# Creating a list of stop words so those are excluded from our all_words list
# The parameter for stopwords.words() needs to be the language we use
stop_words = stopwords.words("english")

# Adding some more custom stop words
more_stop_words = ['.', ',', ':', ';', '?', '!', '-', '\"', '\'']
stop_words += more_stop_words # Adding our custom list to the built-in list

# Looping through the words in the 150th entry of 'reviews', same as above
# Getting every word in the first element of the tuple (list of words)
for word in reviews[150][0]:
    # Excluding the stop words as we did in 02_stop_words.py
    if word not in stop_words:
        all_words.append(word.lower()) # better not to distinguish lower and upper

print('\nList of words in 150th review:', all_words)

# Now, we can convert our list of words into an NLTK frequency distribution list
all_words = nltk.FreqDist(all_words)

# Printing the 10 most common words
# This provides a list of tuples (word, frequency)
print('\nMost common words:', all_words.most_common(10))

# Note: we could do the same but just with the 'pos' or 'neg' reviews
# In this example, for the sake of simplicity, only the 150th review was taken into account

# Getting the number of occurrences of a specific word provided by the user
print('\n# of fun:', all_words["fun"])
