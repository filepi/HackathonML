import numpy as np  
import pandas as pd  
import re  
import nltk  
#import matplotlib.pyplot as plt  
import csv
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')

english_stop_words = stopwords.words('english') 


def cleanData(review_texts):
  
  processed_reviews = []
  for sentence in range(0, len(review_texts)):
      processed_feature = re.sub(r'\W', ' ', str(review_texts[sentence])) #remove special characters
      processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature) #remove all single .Ex: If we remove the ' from jack's, a s will remain alone.
      processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) #remove single characters from the start
      processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I) #replace multiple spaces for single spaces
      processed_feature = re.sub(r'^b\s+', ' ', processed_feature) #remove prefixed b
      processed_feature = processed_feature.lower()
  
      processed_feature = lematizeSentences(processed_feature)
      processed_reviews.append(processed_feature)
      
  return processed_reviews


def stemSentences(sentence, algorithm):
  token_words=word_tokenize(sentence)
  stem_sentence=[]
    
  for word in token_words:
    if algorithm == "PorterStemmer":
      stem_sentence.append(PorterStemmer().stem(word))
      stem_sentence.append(" ")
    elif algorithm == "LancasterStemmer":
      stem_sentence.append(LancasterStemmer().stem(word))
      stem_sentence.append(" ")

  return "".join(stem_sentence)  
  

def is_noun(tag):
  return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
  return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
  return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
  return tag in ['JJ', 'JJR', 'JJS']
  
def lematizeSentences(sentence):
  word_lemmetizer = WordNetLemmatizer();

  token_words=word_tokenize(sentence)
  words_by_class = nltk.pos_tag(token_words)
    
  stem_sentence=[]
  for word_by_class in words_by_class:
      if is_noun(word_by_class[1]) == True:
        stem_sentence.append(word_lemmetizer.lemmatize(word_by_class[0], pos="n"))
      elif is_verb(word_by_class[1]) == True:
        stem_sentence.append(word_lemmetizer.lemmatize(word_by_class[0], pos="v"))
      elif is_adverb(word_by_class[1]) == True:
        stem_sentence.append(word_lemmetizer.lemmatize(word_by_class[0], pos="r"))
      elif is_adjective(word_by_class[1]) == True:
        stem_sentence.append(word_lemmetizer.lemmatize(word_by_class[0], pos="a"))
      else:
        stem_sentence.append(word_lemmetizer.lemmatize(word_by_class[0], pos="s"))

      stem_sentence.append(" ")
  return "".join(stem_sentence)
  def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)
  
def remove_stop_words(review_texts):
  removed_stop_words = []
  for review in review_texts:
    removed_stop_words.append(' '.join([word for word in review.split() 
                      if word not in english_stop_words]))
  return removed_stop_words

def get_lemmatized_text(review_texts):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in review_texts]

