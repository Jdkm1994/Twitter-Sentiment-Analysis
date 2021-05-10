#!/usr/bin/env python
# coding: utf-8

# # TWITTER SENTIMENT ANALYSIS PROJECT USING TEXT MINING CONCEPTS
# 
# 
# <b> Text Mining </b> :- Is nothing but how we analyze the Text to extract "Meaningful Information" out of it.
#     
# <b> What is Text Mining Analysis ? </b> :- Is a process of deriving high quality information from the "text or unstructured data" or the process of extracting interesting and non trivial information and knowledge from unstructured text.
#     
# At a very High Level there are three types of data available or three types of analysis are available :-
# (1) Data Mining
# (2) Text Mining and 
# (3) Image Mining
# 
# If we take <b>"Data Mining"</b> all the <b>"Structured Data"</b> comes under the <b>"Data Mining Analysis"</b> and All the <b>"Unstructured Data"</b> comes under the </b>"Text Mining Analysis and Image Mining"</b>. 
#     
# Data Mining is nothing but the data which we have collected in the format of <b>[Excel,JSON, Text or CSV etc]</b> and then we pre-processed the data using different pre-processing techniques like <b>[Missing Value Analysis, Outlier Analysis, Feature Selection, Feature Scaling, Normalization, Sampling Techniques, and Data Reduction etc]</b> and then we have developed a model on top of it.
# 
# 
# Suppose if the Data is in the form of "TEXT" means like <b>{Reviews, Feedback, Comments, Emails etc}</b> then how will you deal with that type of data beacuse, models will not accept the <b>"Text Data"</b> as your "Input" and because the models will only accept the data in the form of "Tables". 
# 
# For Example we have a dataset of 10 Independent variables and 1 dependent variable and which is in the form of rows and columns in the form of table. Then how should one deal with the "Text" and how will you convert the text into proper "Structured or Tabular Format" ?. But When we are speaking about "text" we are talking about a "sentence" or a "review" or any "comment" or any "email" is a combination of different "words".
# 
# 
# So Same Concepts as Data Mining has for Data Pre-Processing here we need to do the Pre-Processing of Text using Different Text-Pre-Processing Techniques such as they are removing <b> {Numbers, Punctuation marks, Stop Words etc}</b> and then Converting the unstructured or a "simple sentence" to a <b>Tabular Format</b>.
# 
# <b> It is the process of identifying a novel information from a collection of texts (also known as a corpus)</b>. Which helps us to discover usefull and previously unknown "gems"of information in "large text collections" which are patterns, associations or trends.
#     
# So basically "Text Mining" is nothing but applying your analysis or Machine Learning Algorithms on the text to "extaract the hidden patterns out of it. It will help the organization to drive meaningfull information or valuable business insights from text based on the contents like email, word documents or postings on social media or tweets on the twitter.

# # Problem Statement
# 
# <b> The objective of this Task is to detect "Hate Speech" in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a "racist or sexist tweets from other tweets.<b> 
#     
# <b> I Personally quite like this task because these days hate speech, trolling and social media bullying have become serious issues these days and a system that is able to detect such texts would surely be of great use in making the internet and socail media a better and bully-free place.<b>

# Approach and the Steps Involved for the problem statement :-
# 1. Understand The Problem Statement
# 2. Tweets Preprocessing and Cleaning
# 3. Story Generation and Visualization from Tweets
# 4. Extracting Features from Cleaned Tweets
# 5. Model Building: Sentiment Analysis 
# 
# (1) Formally, given a training sample of tweets and lables, where lablel '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the lables on the given test dataset.
#     
# <b>The evaluation metric used here is F1-Score<b>

# If the data is arranged in a structured format then it becomes easier to find the right information. The pre-processing of the text data is an essential step as it makes the raw text ready for mining i.e; it becomes easier to extract information from text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don't carry much weightage in context to the text.   

# In[1]:


# Importing All Standard Libraries
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Setting the working directory  
os.chdir("F:\DataScienceEdwisor\PythonScripts")

# Checking the current working directory
os.getcwd()

# loading the data
train = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')


# In[3]:


# Lets check the first few rows of the train dataset
train.head()


# <b> 2. Tweets Preprocessing and Cleaning </b>

# The data has 3 columns <b>id, label and tweet</b>. Label is the binary target variable and tweet contains the tweets we will clean and preprocess.

# Initial data cleaning requirements that we can think of after looking at the top 5 records:-
# 
# The Twitter handles are already masked as @user due to privacy concerns. So, these Twitter handles
# are hardly giving any information about the nature of the tweet.
# We can also think of getting rid of the punctuations, numbers and even special characters since they
# wouldn’t help in differentiating different kinds of tweets.
# Most of the smaller words do not add much value. For example, ‘pdx’, ‘his’, ‘all’. So, we will try to
# remove them as well from our data.
# Once we have executed the above three steps, we can split every tweet into individual words or tokens
# which is an essential step in any NLP task.
# In the 4th tweet, there is a word ‘love’. We might also have terms like loves, loving, lovable, etc. in the
# rest of the data. These terms are often used in the same context. If we can reduce them to their root
# word, which is ‘love’, then we can reduce the total number of unique words in our data without losing a
# significant amount of information.

# In[4]:


# (A) Removing Twitter handles (@user)
# FOr convenience, let's first combine train and test set. 
combi = train.append(test, ignore_index=True)


# In[5]:


# Defining user defined function to remove unwanted text patterns from the tweets
def remove_pattern(input_txt, pattern):
    
    """ It takes two arguments, one is the original string of text and the other is the pattern 
        of text that we want to remove from the string. The function returns the same input string 
        but without the given pattern. We will use this function to remove the pattern ‘@user’ from all the tweets in 
        our data. """
    
    r = re.findall(pattern, input_txt)
    
    for i in r:
        
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


# In[6]:


# Now lets create a new column tidy_tweet
# remove twitter hndles (@user)
# Note that we have passed “@[\w]*” as the pattern to the remove_pattern function. It is actually a regular expression which
# will pick any word starting with ‘@’.
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")


# In[7]:


# (B) Removing Punctuations, Numbers, and Special Characters
# Here we will replace everything except characters and hashtags with spaces.
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")


# In[8]:


# (C) Removing Short Words
# We have to be a little careful here in selecting the length of the words which we want to remove. So, I have decided to 
# remove all the words having length 3 or less. For example, terms like “hmm”, “oh” are of very little use. 
# It is better to get rid of them.
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# In[9]:


# Lets take another look at the first few rows of the cobined dataframe
combi.head()


# we can clearly observe the difference between the raw tweets and the cleaned tweets (tidy_tweet) quite clearly. only the important word in the tweets have been retained and the noise (numbers, punctuations and special characters) has been removed.

# In[10]:


# (D) Tokenization:- Now we will tokenize all the cleaned tweets in our dataset. 
# Tokens are individual terms or word, and tokenization is the process of splitting
# a string of text into tokens.
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())

tokenized_tweet.head()


# In[11]:


# (E) Stemming:- Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word. 
# For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”.
from nltk.stem.porter import *

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()


# In[12]:


# Now stitch these tokens back togeather.
for i in range(len(tokenized_tweet)):
    
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    
combi['tidy_tweet'] = tokenized_tweet


# <b> 3. Story Generation and Visualization from Tweets </b>

# we will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights.
# 
# Before we begin exploration, we must think and ask questions related to the data in hand. A few probable questions are as follows:
# What are the most common words in the entire dataset?
# What are the most common words in the dataset for negative and positive tweets, respectively?
# How many hashtags are there in a tweet?
# Which trends are associated with my dataset?
# Which trends are associated with either of the sentiments? Are they compatible with the sentiments?

# In[13]:


# (A) Understanding the common words used in the tweets: WordCloud
# Now I want to see how well the given sentiments are distributed across the train dataset. One way to accomplish this 
# task is by understanding the common words by plotting wordclouds.

# A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.
# Let’s visualize all the words our data using the wordcloud plot.
all_words = ' '.join([text for text in combi['tidy_tweet']]) 

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()


# We can see most of the words are positive or neutral. With happy and love being the most frequent ones. It doesn’t give us any idea about the words associated with the racist/sexist tweets. Hence, we will plot separate wordclouds for both the classes(racist/sexist or not) in our train data.

# In[14]:


# (B) Words in non racist/sexist tweets
normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()


# We can see most of the words are positive or neutral. With happy, smile, and love being the most frequent ones. Hence, most of the frequent words are compatible with the sentiment which is non racist/sexists tweets. Similarly, we will plot the word cloud for the other sentiment. Expect to see negative, racist, and sexist terms.

# In[15]:


# (C) Racist/Sexist Tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])

wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()


# As we can clearly see, most of the words have negative connotations. So, it seems we have a pretty good text data to work on. Next we will the hashtags/trends in our twitter data.

# In[16]:


# (D) Understanding the impact of Hashtags on tweets sentiment

# Hashtags in twitter are synonymous with the ongoing trends on twitter at any particular point in time. We
# should try to check whether these hashtags add any value to our sentiment analysis task, i.e., they help in
# distinguishing tweets into the different sentiments.

# function to collect hashtags
def hashtag_extract(x):
    
    hashtags = []
    
    # loop over the words in the tweet
    
    for i in x:
        
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
        
    return hashtags


# In[17]:


# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

# extracting hashtags from racist/sexist tweets

HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# Now that we have prepared our lists of hashtags for both the sentiments, we can plot the top n hashtags. So, first let’s check the hashtags in the non racist/sexist tweets.

# In[18]:


# Non-Racist/Sexist Tweets
a = nltk.FreqDist(HT_regular)

d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags
d = d.nlargest(columns="Count", n = 10)

plt.figure(figsize=(16,5))

ax = sns.barplot(data=d, x="Hashtag", y ="Count")

ax.set(ylabel = 'Count')

plt.show()


# All these hashtags are positive and it makes sense. I am expecting negative terms in the plot of the second list. Let’s check the most frequent hashtags appearing in the racist/sexist tweets.

# In[19]:


# Racist/Sexist Tweets

b = nltk.FreqDist(HT_negative)

e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)

plt.figure(figsize=(16,5))

ax = sns.barplot(data=e, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()


# As expected, most of the terms are negative with a few neutral terms as well. So, it’s not a bad idea to keep these hashtags in our data as they contain useful information. Next, we will try to extract features from the
# tokenized tweets.

# <b> 4. Extracting Features from Cleaned Tweets Bag-Of-Words</b>
# 
# To analyze a preprocessed data, it needs to be converted into features. Depending upon the usage, text
# features can be constructed using assorted techniques – Bag-of-Words, TF-IDF, and Word Embeddings. In
# this article, we will be covering only Bag-of-Words and TF-IDF.

# In[20]:


# Bag-of-Words Features
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])


# <b>TF -IDF Features</b> 
# 
# This is another method which is based on the frequency method but it is different to the bag-of-words approach in the sense that it takes into account, not just the occurrence of a word in a single document (or tweet) but in the entire corpus.
# 
# TF-IDF works by penalizing the common words by assigning them lower weights while giving importance to words which are rare in the entire corpus but appear in good numbers in few documents. Let’s have a look at the important terms related to TF-IDF:
# 
# TF = (Number of times term t appears in a document)/(Number of terms in the document)
# IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has
# appeared in.
# TF-IDF = TF*IDF

# In[21]:


# TF-IDF Fetaures
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectoirzer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# TF-IDF Feature Matrix
tfidf = tfidf_vectoirzer.fit_transform(combi['tidy_tweet'])


# <b> 5. Model Building: Sentiment Analysis </b>

# We are now done with all the pre-modeling stages required to get the data in the proper form and shape. Now
# we will be building predictive models on the dataset using the two feature set — Bag-of-Words and TF-IDF.
# We will use logistic regression to build the models. It predicts the probability of occurrence of an event by
# fitting data to a logit function.

# In[23]:


# (A) Building model using Bag-of-Words features

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,-1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0

prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1-score


# We trained the logistic regression model on the Bag-of-Words features and it gave us an F1-score of 0.53 for
# the validation set. Now we will use this model to predict for the test data.

# In[24]:


test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file


# In[25]:


# (B) Building model using TF-IDF features
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)


# In[ ]:




