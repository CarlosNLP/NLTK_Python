from nltk.corpus import wordnet

# This gives back the list of sinsets for 'dog', like Synset('dog.n.01')
print(wordnet.synsets('dog'))

# Using the synset to retrieve and print its 'definition'
dogDefinition = wordnet.synset('dog.n.01').definition()
print('Definition:', dogDefinition)

# Using the 'examples' method to get a list of sentences with that word
print(wordnet.synset('dog.n.01').examples())

# Getting just the word itself and not the whole 'dog.n.01'
# Note: both synsets and lemmas are lists, so we access the first element [0]
dog = wordnet.synsets('dog')[0].lemmas()[0].name()
print(dog)

# Working with semantic similatiry between words
# Note: similarity is count between 0 (less similar) and 1 (equal)
print('Printing semantic similarities between words...')
boat = wordnet.synset('boat.n.01')
ship = wordnet.synset('ship.n.01')

similarity1 = boat.wup_similarity(ship)
print('boat vs ship:', similarity1)

# Now two different words semantically
book = wordnet.synset('book.n.01')
sheep = wordnet.synset('sheep.n.01')

similarity2 = book.wup_similarity(sheep)
print('book vs sheep:', similarity2)
