import nltk

sentence = """The group arrived at two o'clock on Monday afternoon to start
class."""

# Tokenizing
tokens = nltk.word_tokenize(sentence)
print "Tokens:"
print tokens
print

# POS tagging
tagged = nltk.pos_tag(tokens)
print "Tags:"
print tagged

