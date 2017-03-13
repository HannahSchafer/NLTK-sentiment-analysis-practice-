import nltk
from nltk.tokenize import PunktSentenceTokenizer

sentence1 = """The group arrived at two o'clock on Monday afternoon to start
class."""
sentence2 = """"The Little Mermaid" (Danish: Den lille havfrue) is a fairy tale 
written by the Danish author Hans Christian Andersen about a young mermaid who 
is willing to give up her life in the sea and her identity as a mermaid to gain 
a human soul."""

#Chunking
custom_sent_tokenizer = PunktSentenceTokenizer(sentence1)
tokenized = custom_sent_tokenizer.tokenize(sentence2)

def process_content():
    for i in tokenized:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)
        # chunked.draw() 
        print chunked

process_content()




# many named nouns
# chunking: chunk: 'noun phrases' be anoun, and modifiers around that noun.
# descriptive group of words surrounding that noun. downside: only can use
# regular expressions. can only chunk things that are touching each other (downside)
# can chunk important words and break it out from there.