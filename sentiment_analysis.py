# -*- coding: utf-8 -*-

import nltk
from nltk.probability import FreqDist
from nltk.classify.util import apply_features,accuracy

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    """Get frequency of occurence for each word"""
    wordlist = FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


pos_tweets=[('I love this car','positive'), 
('This view is amazing','positive'),
('I feel great this morning','positive'),
('I am so excited about the concert','positive'),
('He is my best friend','positive')]

neg_tweets=[('I do not like this car','negative'),
('This view is horrible','negative'),
('I feel tired this morning','negative'),
('I am not looking forward to the concert','negative'),
('He is my enemy','negative')]

"""Take both of those lists and create a single list of tuples each containing 
two elements. First element is a list containing the words and second element 
is the type of sentiment. Get rid of the words smaller than 2 characters and 
we use lowercase for everything."""
tweets=[]
for(words,sentiment)in pos_tweets+neg_tweets:
    words_filtered=[e.lower() for e in words.split() if len(e)>=3]
    tweets.append((words_filtered,sentiment))

"""Create test tweet list"""
test_pos_tweets=[('I feel happy this morning','positive'), 
('Larry is my friend','positive')]

test_neg_tweets=[('I do not like that man','negative'),
('This view is horrible','negative'),
('The house is not great','negative'),
('Your song is annoying','negative')]

test_tweets=[]
for(test_words,test_sentiment)in test_pos_tweets+test_neg_tweets:
  test_words_filtered=[e.lower() for e in test_words.split() if len(e)>=3]
  test_tweets.append((test_words_filtered,test_sentiment))

"""CLASSIFIER"""
"""Use functions at the top to find occurence of each word, save to word_features"""
word_features = get_word_features(get_words_in_tweets(tweets))

"""Classifier will decide which tweets have which sentiment. To create classifier, 
need to decide what features are relevant. First need feature extractor. This 
one returns a dictionary indicating what words are contained in the input passed. 
Here, the input is the tweet. Use the word features list defined above along 
with the input to create the dictionary."""
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features       

"""With the feature extractor, apply the features to the classifier using the 
method apply_features. Pass the feature extractor along with the tweets list."""
training_set = apply_features(extract_features, tweets)

"""The variable training_set contains the labeled feature sets. It is a list 
of tuples with each tuple containing the feature dictionary and the sentiment 
string for each tweet."""
test_training_set=apply_features(extract_features, test_tweets)

"""Use the training set to train the classifier. Classifier determines
the probability of a particular sentiment. Naive Bayes classifier uses the 
prior probability of each label which is the frequency of each label in the 
training set, and the contribution from each feature. In our case, the 
frequency of each label is the same for positive and negative. The word 
amazing appears in 1 of 5 of the positive tweets and none of the negative tweets. 
This means that the likelihood of the positive label will be multiplied by 0.2 
when this word is seen as part of the input."""
classifier = nltk.classify.NaiveBayesClassifier.train(training_set)

tweet1 = 'Larry is my friend'
print tweet1
print classifier.classify(extract_features(tweet1.split()))
print "----------------------------------------------------"

tweet2 = 'My house is not great'
print tweet2
print classifier.classify(extract_features(tweet2.split()))
print "----------------------------------------------------"

"""For tweet2, the word ‘great’ weights more on the positive side but the 
word ‘not’ is part of two negative tweets in our training set so the output 
from the classifier is ‘negative’. Of course, the following tweet: 
‘The movie is not bad’ would return ‘negative’ even if it is ‘positive’. 
Again, a large and well chosen sample will help with the accuracy of the classifier."""

tweet3 = 'Your song is annoying'
print tweet3
print classifier.classify(extract_features(tweet3.split()))
print "----------------------------------------------------"

"""The classifier thinks it is positive. The reason is that we don’t have any 
information on the feature name ‘annoying’. Larger the training sample tweets 
is, better the classifier will be."""

# """Testing accuracy of the classifier"""
# classifier.show_most_informative_features(5)
# print "----------------------------------------------------"
# print "Accuracy/quality of this classifier:"
# print nltk.classify.util.accuracy(classifier, test_training_set)

# print test_training_set
# print training_set


