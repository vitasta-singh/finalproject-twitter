Objective:  
The following project is about analyzing the sentiments of tweets on social networking website ‘Twitter’. The dataset for this project  is scraped from Twitter. It contains 1,600,000 tweets extracted using Twitter API. It is a labeled dataset with tweets annotated with the sentiment (0 = negative, 2 = neutral, 4 = positive).
It contains the following 6 fields:
target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
ids: The id of the tweet .
date: The date of the tweet (Sat May 16 23:58:44 UTC 2009)
flag: The query. If there is no query, then this value is NO_QUERY.
user: The user that tweeted 
text: The text of the tweet.
Design a classification model that correctly predicts the polarity of the tweets provided in the dataset.  
  
Libraries :  
streamlit, numpy, pandas, seaborn, matplotlib, regularexpression, wordcloud, nltk, sklearn model selection, Tfidvectorizer, pickel.  
  
Procedure:  
1) Load the dataset using twitter api and convert into csv file.
2) Extract the columns having text and target(0,1).
3) Check the ration for positive and negative comments.
4) concat the pos and neg data. CLean the data and check for null values.
5) Now convert into lower case, remove punctuation, special character, URL, stopwords, repetive text.
6) Apply tokenization for splitting the sentence into comma separated text.
7) Apply Portstemmer and lemitization.
8) Use tfidf vectorization for occurence of relevent text in the statement.
9) SPlit the data into training and testing apply ML model logistic regression, SVM, bernouilli naive and choose the model having highest accuracy.
10) Save the model using pickel and test it for new data..
 
