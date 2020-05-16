#! /usr/bin/env python

import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer

# 31/03/2020 Building a Simple Chatbot from Scratch in Python (using NLTK)
GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up","hey"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

#WordNet is a semantically-oriented dictionary of English included in NLTK.
lemmer = nltk.stem.WordNetLemmatizer()

# string.punctuation returns all punctuation values
# ord(punct) returns the integer representing the Unicode code point of that character
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def main():

    # file that contains corpus
    f=open('wiki_cambridge.txt','r', encoding='utf8',errors = 'ignore')

    raw = f.read()
    raw = raw.lower()# converts to lowercase

    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
    word_tokens = nltk.word_tokenize(raw)# converts to list of words

    flag = True

    print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type bye!")

    while(flag == True):
        user_response = input()
        user_response = user_response.lower()
        if(user_response != 'bye'):
            if(user_response == 'thanks' or user_response == 'thank you'):
                flag = False
                print("ROBO: You are welcome..")
            else:
                if(greeting(user_response) != None):
                    print("ROBO: " + greeting(user_response))
                else:
                    print("ROBO: ", end = "")
                    print(response(user_response, sent_tokens, lemmer, remove_punct_dict))
                    sent_tokens.remove(user_response) # ?
        else:
            flag = False
            print("ROBO: Bye! take care..")

    # lemmanize tokens (a variant of stemming)
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

    # normalise text
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response, sent_tokens, lemmer, remove_punct_dict):

    robo_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        robo_response=robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

# if greeting is found in list, return random reply greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


if __name__ == "__main__":
    main()
