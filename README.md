# Sentiment-Analysis-on-tweets


A Twitter sentiment analysis determines negative, positive, or neutral emotions within the text of a tweet using NLP and ML models. Sentiment analysis or opinion mining refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of people on social media for a variety of topics.

# What is Twitter Sentiment Analysis?
Twitter sentiment analysis analyzes the sentiment or emotion of tweets. It uses natural language processing and machine learning algorithms to classify tweets automatically as positive, negative, or neutral based on their content. It can be done for individual tweets or a larger dataset related to a particular topic or event.

# Dataset Used

# SMILE Twitter Emotion Dataset
This dataset is collected and annotated for the SMILE project http://www.culturesmile.org. This collection of tweets mentioning 13 Twitter handles associated with British museums was gathered between May 2013 and June 2015. It was created for the purpose of classifying emotions, expressed on Twitter towards arts and cultural experiences in museums.
It contains 3,085 tweets, with 5 emotions namely anger, disgust, happiness, surprise and sadness. 

# Sentiment140 dataset with 1.6 million tweets
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment.
# Content
It contains the following 6 fields:
target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
ids: The id of the tweet ( 2087)
date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
flag: The query (lyx). If there is no query, then this value is NO_QUERY.
user: the user that tweeted (robotickilldozr)
text: the text of the tweet (Lyx is cool)


# Objective - 1 (Two Classical Machine learning models such as Naive bayes, SVM, Decision tree etc.)

# Random Forest Algorithim
A Random Forest Algorithm is a supervised machine learning algorithm that is extremely popular and is used for Classification and Regression problems in Machine Learning. We know that a forest comprises numerous trees, and the more trees more it will be robust. Similarly, the greater the number of trees in a Random Forest Algorithm, the higher its accuracy and problem-solving ability.  Random Forest is a classifier that contains several decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset. It is based on the concept of ensemble learning which is a process of combining multiple classifiers to solve a complex problem and improve the performance of the model.

# Logistic Regression
Decision trees are a popular machine learning algorithm that can be used for both regression and classification tasks. They are easy to understand, interpret, and implement, making them an ideal choice for beginners in the field of machine learning. It is a tool that has applications spanning several different areas. Decision trees can be used for classification as well as regression problems. The name itself suggests that it uses a flowchart like a tree structure to show the predictions that result from a series of feature-based splits. It starts with a root node and ends with a decision made by leaves.

# Objective - 2 (BI-LSTM with word2vec/ Fasttext word embedding)

# word2vec
Word2vec is a technique for natural language processing (NLP) published in 2013. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence.

# Fasttext
FastText is a word embedding technique that provides embedding to the character n-grams. It is the extension of the word2vec model. FastText is an open-source library, developed by the Facebook AI Research lab. Its main focus is on achieving scalable solutions for the tasks of text classification and representation while processing large datasets quickly and accurately. FastText is a modified version of word2vec.

# How FastText is Better

It treats each word as composed of n-grams. In word2vec each word is represented as a bag of words but in FastText each word is represented as a bag of character n-gram.


# Objective - 3 (Transformer based model with BERT-based word embedding)

# BERT
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model based on the Transformer architecture. It has achieved state-of-the-art performance on various natural language processing (NLP) tasks, such as question answering, text classification, and named entity recognition.
BERT employs a technique called "word embeddings" to represent words in a dense vector space. Word embeddings capture semantic and syntactic information, allowing the model to understand the meaning and relationships between words. BERT uses a contextualized word embedding approach, meaning that the word representation depends on its context in the sentence.


# Comparitive Study 
In the comparitive study i have compare the analysis obtained by various models used in sentiment analysis.



