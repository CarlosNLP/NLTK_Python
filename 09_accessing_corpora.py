# Importing modules with datasets within nltk.corpus
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import nps_chat
from nltk.corpus import brown
from nltk.corpus import reuters

# Printing the list of all dataset names in each module
print('Printing the file IDs for each module...\n')
print('gutenberg:\n', gutenberg.fileids())
print('webtext:\n', webtext.fileids())
print('nps_chat:\n', nps_chat.fileids())
print('brown:\n', brown.fileids())
print('reuters:\n', reuters.fileids())

# Printing the categories of each module
# NOTE: gutenberg, webtext and nps_text do not have "categories"
print('Printing the categories for each module, if available...\n')
print('brown:\n', brown.categories())
print('reuters:\n', reuters.categories())

# Accessing the corpora
# NOTE: TXT files can be accessed through "raw" to get the full files
print('Accessing the sample files...')
print('gutenberg:\n', gutenberg.raw("austen-emma.txt"))

# Accessing sentences of a sample file
print('Getting a list of sentences...')
print('List of sentences from austen-emma.txt:\n', gutenberg.sents("austen-emma.txt"))
print('List of sentences from a chat:\n', nps_chat.posts("10-19-20s_706posts.xml"))

# Example going through each post from a chat
posts = nps_chat.posts("10-19-20s_706posts.xml")

for post in posts:
    print('- ', post)

# As we can see, 'posts' is a list of all the posts in the file 10-19-20s_706posts.xml.
# Each entry from that list is also a list of words for each post.
