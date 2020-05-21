# Importing the packages we will use in the script
import nltk # we use nltk.pos_tag() later on. Note: 'pos' stands for Part of Speech
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

# Ref: PunktSentenceTokenizer
# https://nlpforhackers.io/splitting-text-into-sentences/

# NOTE
# PunktSentenceTokenizer uses an unsupervised and pre-trained algorithm
# It will used the pre-trained version by default if empty initialized, but
# if initialized with some text, it will be trained too with that content

# Loading the training text that will be used to initialize the algorithm
train_text = state_union.raw("2005-GWBush.txt")

# Loading the sample text that will be used for the POS tagging
sample_text = state_union.raw("2006-GWBush.txt")

# Initializing PunktSentenceTokenizer() with the training text
# It will give back an object of PunktSentenceTokenizer class
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# This gives back a list of sentences detected by the algorithm
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Printing all of them for reference
print(tokenized)

# Running through all the items from the list (i.e.: all the detected sentences)
for i in tokenized:
    # Creating a new list with all the words from every current sentece
    words = word_tokenize(i)
    # Getting the POS (part of speech) of every word from the current sentence
    # NOTE: this gives back a list of tuples ('word', 'POS')
    tagged = nltk.pos_tag(words)
    print(tagged)

# For reference, a list of meanings of the POS tags can be found here:
# https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
