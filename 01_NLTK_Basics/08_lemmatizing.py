# Importing the packages that we will need in this example
from nltk.stem import WordNetLemmatizer
import nltk

# Creating an object of the WordNetLemmatizer class
lemmatizer = WordNetLemmatizer()

# Making a list of sample words
my_words = ["played", "phones", "helping", "better", "lives"]

# Printing how the default lemmatization would be for these words
print('Printing results with default lemmatizer (i.e.: noun as default POS)')

# Going through each word and lemmatize
for word in my_words:
    print(word, 'lemmatized as', lemmatizer.lemmatize(word))

# This works OK with nouns because the default 'pos' used is a noun

# Retrieving POS tagging from the list of words
tagged = nltk.pos_tag(my_words) # this creates a list of tuples (word, 'POS')
print('\n', tagged) # printing the POS tagging just for reference

# With WordNetLemmatizer we could specify the POS tag for each word with pos=""
# Printing how the words are lemmatized with WordNetLemmatizer for each POS
print('\nPrinting results with customized lemmatizer per POS tag')

# Going through each word and lemmatize with the proper POS tag
# NOTE: here I use range(len(my_words)) to be able to detect the index from the list too, with the corresponding word and POS tag
for i in range(len(my_words)):
    if tagged[i][1].startswith("V"): # this works for VBN, VBG...
        print(my_words[i], 'lemmatized as', lemmatizer.lemmatize(my_words[i], pos="v")) # pos="v" means verb
    elif tagged[i][1].startswith("J") or tagged[i][1] == "RBR" or tagged[i][1] == "RBS": # adjectives or adverbs for comparative or superlative
        print(my_words[i], 'lemmatized as', lemmatizer.lemmatize(my_words[i], pos="a")) # pos="a" means adjective
    else: # all the others will use noun which is the default, this works in this example because I haven't added any other POS type
        print(my_words[i], 'lemmatized as', lemmatizer.lemmatize(my_words[i]))

# For reference, a list of meanings of the POS tags can be found here:
# https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
