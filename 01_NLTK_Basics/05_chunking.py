# Importing the packages we will use in the script
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

# Loading the sample text that will be used for the POS tagging
sample_text = gutenberg.raw("bible-kjv.txt")

# Empty initializing PunktSentenceTokenizer()
# This will use the pre-trained algorithm by default
# It will give back an object of PunktSentenceTokenizer class
custom_sent_tokenizer = PunktSentenceTokenizer()

# This gives back a list of sentences detected by the algorithm
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Running through the first 20 sentences
for i in tokenized[:20]:
    # Creating a new list with all the words from every current sentece
    words = word_tokenize(i)
    # Getting the POS (part of speech) of every word from the current sentence
    # NOTE: this gives back a list of tuples ('word', 'POS')
    tagged = nltk.pos_tag(words)
    # Defining the chunk pattern with regular expressions (if needed)
    # We'll be looking for preposition + determiner (optional) + adjective (zero or more) + noun
    chunkGram = r"Prepositional phrase: {<IN><DT>?<JJ>*<NN>}"
    # Creating the parser with the specified pattern
    chunkParser = nltk.RegexpParser(chunkGram)
    # Parsing the list of tuples ('word', 'POS')
    chunked = chunkParser.parse(tagged)
    # Draw the result to pretty visualize it
    # NOTE: the draw will be displayed in a separate window
    chunked.draw()

# Feel free to experiment yourself!

# For reference, a list of meanings of the POS tags can be found here:
# https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
