# Twitter-Sentiment-Analysis
Given Twitter US Airline Sentiment Dataset, which contains data for over 14000 tweets,  task is to predict the sentiment of the tweet i.e. positive, negative or neutral.
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk import pos_tag
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree