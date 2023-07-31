# Twitter-Sentiment-Analysis
#Given Twitter US Airline Sentiment Dataset, which contains data for over 14000 tweets,  task is to predict the sentiment of the tweet i.e. positive, negative or neutral.
## Importing the Required Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np

training_data=pd.read_csv("training_twitter_x_y_train.csv")
testing_data=pd.read_csv('test_twitter_x_test.csv')
training_documents=training_data['text'].values
training_categories=training_data['airline_sentiment'].values
testing_documents=testing_data['text'].values

count_vect=TfidfVectorizer(max_features=5000, max_df=0.8, min_df=0.001)
x_train=count_vect.fit_transform(training_documents)
x_test=count_vect.transform(testing_documents)
y_train=training_categories
clf=RandomForestClassifier(n_estimators=2000, n_jobs=-1)
clf.fit(x_train, y_train)
y_test=clf.predict(x_test)

np.savetxt(fname='predicted.csv', X=y_test, fmt='%s')
