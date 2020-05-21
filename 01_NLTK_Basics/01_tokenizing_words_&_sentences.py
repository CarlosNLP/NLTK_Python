# Importing the packages we need: sent_tokenize (for sentences) and word_tokenize (for words/tokens)
from nltk.tokenize import sent_tokenize, word_tokenize

#####################################
# Working with sample text from a file

# Opening and reading our sample text
fh = open('Samples\\bible-kjv.txt', 'r', encoding='utf-8')
content = fh.read()

# NOTE
# sent_tokenize(content) / word_tokenize(content) methods give back Python lists
# The parameter given to these methods must be a string
sentences = sent_tokenize(content)
words = word_tokenize(content)

# Printing the number of sentences and words/tokens
# We use 'len' to get the number of items in our lists
print('\nSAMPLE FROM bible-kjv.txt\n')
print('# of sentences:', len(sentences))
print('# of tokens:', len(words))

# Printing the first 10 words/tokens from the list
for i in range(10):
    print(words[i])

#####################################
# Working with sample text written by us
content = 'Hey there, I\'m Mr. Smith. Nice you e-meet you.'

# NOTE
# sent_tokenize(content) / word_tokenize(content) methods give back Python lists
# The parameter given to these methods must be a string
sentences = sent_tokenize(content)
words = word_tokenize(content)

# Printing the number of sentences and words/tokens
# We use 'len' to get the number of items in our lists
print('\nSAMPLE FROM OUR TEXT\n')
print('# of sentences:', len(sentences))
print('# of tokens:', len(words))

# Printing all the words/tokens from our sampe text
for i in words:
    print(i)
