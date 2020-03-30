import nltk
import numpy as np
import random
import string # to process standard python strings

# 31/03/2020 Building a Simple Chatbot from Scratch in Python (using NLTK)

f=open('wiki_cambridge.txt','r',errors = 'ignore')

raw=f.read()
raw=raw.lower()# converts to lowercase

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words

print(sent_tokens[:2])
print(word_tokens[:2])
