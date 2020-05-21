# Importing the packages we will need for this part
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Working with sample text from a file

# Opening and reading our sample text
fh = open('Samples\\shakespeare-hamlet.txt', 'r', encoding='utf-8')
content = fh.read()

# The parameter for stopwords.words() needs to be the language we use
stop_words = stopwords.words("english")

# We use the word_tokenize function as we used it in the first part of the repo
# We could use set(word_tokenize(content)) and it would be a bit quicker, but
# for the sake of consistency we will use the default type, which is a list
words = word_tokenize(content)

print('\nSAMPLE FROM shakespeare-hamlet.txt\n')

# Printing the number of total tokens (from word_tokenize())
print('# of total tokens:', len(words))
# Printing the number of stop words from the built-in function from nltk.corpus
print('# of stop words:', len(stop_words))
# Printing all the built-in stop words as reference
print(stop_words)

# Our next goal is to get the number of filtered words, which is the number
# of total tokens except for those which are found in the stop words list

# Creating the list quickly 
filtered = [w for w in words if w not in stop_words]

# Same stuff can be expanded as below and we would get the same result
##filtered = []
##for w in words:
##    if w not in stop_words:
##        filtered.append(w)

# Printing the number of filtered words / tokens
# We can see that a huge number of the total tokens are stop words
print('# of filtered tokens:', len(filtered))


###############################
# Appending more stop words, actually some punctuation marks
more_stop_words = ['.', ',', ':', ';', '?', '!', '-']
stop_words += more_stop_words # Adding our custom list to the built-in list
print('# of stop words (updated):', len(stop_words))

# Printing all the built-in stop words + new stop words as reference
print(stop_words)

# New filtered list taking into account the new stop words
filtered = [w for w in words if w not in stop_words]

# Printing the number of filtered words / tokens after updating the list
print('# of filtered tokens (updated):', len(filtered))
