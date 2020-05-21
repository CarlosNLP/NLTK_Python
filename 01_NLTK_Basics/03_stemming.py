# Importing the packages that we will need in this example
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize

# Ref: Stemming vs Lemmatization
# https://www.datacamp.com/community/tutorials/stemming-lemmatization-python

# Creating an object of the PorterStemmer class
ps = PorterStemmer()
print('\nUsing PorterStemmer()\n')

# Making a list of sample words
my_words = ["accomplished", "successful", "helping", "protective", "lives"]

# Printing how the words are stemmed with PorterStemmer
for my_word in my_words:
    print(my_word, 'stemmed as', ps.stem(my_word))

# Same with other set of words
new_words = ["connect", "connections", "connected", "connecting", "connection"]
for new_word in new_words:
    print(new_word, 'stemmed as', ps.stem(new_word))

################################################

# Creating an object of the LancasterStemmer class
ls = LancasterStemmer()
print('\nUsing LancasterStemmer()\n')

# Making a list of sample words (same as before)
my_words = ["accomplished", "successful", "helping", "protective", "lives"]

# Printing how the words are stemmed with LancasterStemmer
for my_word in my_words:
    print(my_word, 'stemmed as', ls.stem(my_word))

# Same with other set of words (same as before)
new_words = ["connect", "connections", "connected", "connecting", "connection"]
for new_word in new_words:
    print(new_word, 'stemmed as', ls.stem(new_word))


################################################

# We will do the same now but going through a list of tokens from a file
# Opening and reading our sample text
fh = open('Samples\\austen-sense.txt', 'r', encoding='utf-8')
content = fh.read()

# word_tokenize(content) method gives back a Python list with all the tokens
# The parameter given to 'word_tokenize' must be a string
words = word_tokenize(content)

print('\nPrinting stemmed tokens from our sample file\n')
# Going through the first 100 entries from the list with PorterStemmer
for word in words[:100]:
    print(word, 'stemmed as', ps.stem(word))
