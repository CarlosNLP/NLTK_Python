# Importing the packages we will use in the script
import nltk
from nltk.corpus import reuters
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

# Loading the sample text that will be used for the POS tagging
sample_text = reuters.raw("test/14982")

# Empty initializing PunktSentenceTokenizer()
# This will use the pre-trained algorithm by default
# It will give back an object of PunktSentenceTokenizer class
custom_sent_tokenizer = PunktSentenceTokenizer()

# This gives back a list of sentences detected by the algorithm
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Running through the sentences
for i in tokenized:
    # Creating a new list with all the words from every current sentece
    words = word_tokenize(i)
    # Getting the POS (part of speech) of every word from the current sentence
    # NOTE: this gives back a list of tuples ('word', 'POS')
    tagged = nltk.pos_tag(words)
    # Chunking for NER - Named-entity recognition
    chunk_NER = nltk.ne_chunk(tagged)
    # Drawing the NER chunk to pretty visualize it
    chunk_NER.draw()
    

# Feel free to experiment yourself!

# For reference, a list of meanings of NER tags can be found here:
# https://www.nltk.org/book/ch07.html - Section 5
