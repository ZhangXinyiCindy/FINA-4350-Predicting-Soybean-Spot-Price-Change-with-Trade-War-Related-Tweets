#!/usr/bin/env python
# coding: utf-8

# # Predicting Soybean Spot Price Change with "Trade War" Related Tweets 
# 
# #### FINA 4350 Text Analytics and Natural Language Processing in Finance and Fintech (2019-20 Semester 1)<a class="tocSkip">
# 
# 
# *Coded by Zhang Xinyi *
# 
# ## Background:  
# 
# Since US-Sino trade war (“Trade War”) burst out in March 2018, the event has affected a wide range of business in the two countries, and the impact even spreads worldwide. Soybean has been among the commodities with largest import amount to China from the U.S., which inevitably bore the brunt of the Trade War. 
# 
# In early April 2018, right after the U.S. published its list of tariffs on Chinese products, Ministry of Commerce of China stroke back by imposing tariffs on 128 products it imports from the U.S., including soybeans with an additional 25% tariff. The high tariff triggered U.S. soybean exports to China. From U.S. Census Bureau, we see that U.S. soybean exports, as well as its percentage bound for China, showed an abnormal shrink after the event. It provides evidence that China plays an important role on U.S. soybean exports. The Trade War and increasing tariff can have a significant impact on this.
# 
# ![CBOT.png](attachment:CBOT.png)
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import re
import eli5
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from collections import Counter
from eli5.sklearn import PermutationImportance
from pandasql import sqldf



import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# ## Import raw tweet data and merge it data with lagged spot price data
# 
# Data source:

# In[2]:


#Import raw twitter data and merge Twitter data
df = pd.read_csv('./data-streaming-tweets.csv')
#Convert to datetime64 and convert UTC time used by Twitter to Eastern Time (New York).
df.date = pd.to_datetime(df.date) - np.timedelta64(5, 'h')
df.date = df.date.dt.date     # Remove time part, only keep date.
#remove data for 01-Dec-2019 as we cutoff at the end of Nov
df = df[df['date']!= df.iloc[3324600]['date']]

#Import Spot price information
SpotPrice = pd.read_csv('./ChangeDateNew.csv')
SpotPrice.date = pd.to_datetime(SpotPrice.date)
SpotPrice.date = SpotPrice.date.dt.date  
SpotPrice['Lag_PriceChange'] = SpotPrice.Price_Change.shift(-1) # Lag the price change

#Merge Spot price with raw data
df = pd.merge(df, SpotPrice, how='left',on='date')

df.tail()                                    


# In[3]:


#data inspection
DailyPosts=df.groupby(['date']).size()
DailyPosts.to_csv(r'/Users/zhangxinyi/Desktop/DailyPosts.csv')


# ## Filter trade war related tweets and cut the "Lagged price change" into quantiles of low, medium, high (lagged) price change.
# 
# 

# In[4]:


#Filter related tweets
TradewarDf=df[df['tweet'].str.contains('trade war')|df['tweet'].str.contains('tariff')]
TradewarDf=TradewarDf[~TradewarDf['tweet'].str.contains('mexic')]
TradewarDf.reset_index(drop=True,inplace=True)

# Cut into quantiles of low, medium, high (lagged) returns.
TradewarDf['q_Lag_PriceChange'] = pd.qcut(TradewarDf.Lag_PriceChange, 3, labels=False)

TradewarDf.head()


# ## Text Preprocessing

# In[5]:


#remove punctuation 

def remove_punctuation (text):
    no_punct=''.join([c for c in text if c not in string.punctuation])
    return no_punct

stop_words_lst = ['$spx', '$spy', '$es','$dji', '$djia', '$indu', '$ym','$nq', '$nasdaq', '$qqq','$tlt',
        '$gc', '$gld',
        '$ng', '$wti']

def remove_key (text):
    no_key=' '.join([c for c in text.split(' ') if c not in stop_words_lst])
    return no_key



lemmatizer = WordNetLemmatizer()
#Instantiate Lemmatizer
def word_lemmatizer (text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

#Instantiate Stemmer
stemmer = PorterStemmer()
def word_stemmer(text):
    stem_text = ''.join([stemmer.stem(i) for i in text])
    return stem_text

#Remove Non-English words
words = set(nltk.corpus.words.words())

def remove_non_English_word(text):
    english_text = ''.join(w for w in nltk.wordpunct_tokenize(text)          if w.lower() in words or not w.isalpha())
    return english_text
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')


def remove_website_links(text):

    no_website_links =text.replace('{html}',"") 
    no_website_links = re.sub(r'https?:\/\/.*[\r\n]*', '', no_website_links, flags=re.MULTILINE)

    return no_website_links
def remove_non_alphabetic_word(text):
    rem_num = re.sub('[0-9]+', '', text)
    return rem_num



#TradewarDf['tweet']=TradewarDf['tweet'].apply (lambda x: remove_punctuation(x))
#remove punctuation is skipped since sklearn take care of removing punctuation when tokenizing
TradewarDf['tweet']=TradewarDf['tweet'].apply (lambda x: remove_key(x))
TradewarDf['tweet']=TradewarDf['tweet'].apply (lambda x: remove_website_links(x))
TradewarDf['tweet']=TradewarDf['tweet'].apply (lambda x: deEmojify(x))
TradewarDf['tweet']=TradewarDf['tweet'].apply (lambda x: remove_non_alphabetic_word(x))
TradewarDf['tweet']=TradewarDf['tweet'].apply (lambda x: word_lemmatizer(x))
TradewarDf['tweet']=TradewarDf['tweet'].apply (lambda x: word_stemmer(x))
TradewarDf['tweet'][5]


# ## Split the dataset & Convert Text to Numbers ( tf-idf )
# Split of training/testing data on 1-11- 2019. construct tfidf(training) and tfidfTesting.  
# Drop words that occurred in more than 90% of of the documents and those appears less than 20 times (roughly 0.05%)

# In[6]:


#TradewarDf.iloc[36900]['date'] is 2019-11-01, 
#the date data type here is unfamiliar to the me so I failed to hard code the date
Training=TradewarDf[TradewarDf['date']<TradewarDf.iloc[36900]['date']]
Testing=TradewarDf[TradewarDf['date']>=TradewarDf.iloc[36900]['date']]
Testing.reset_index(drop=True,inplace=True)

#fit tf-idf with trainning data
v = TfidfVectorizer(stop_words='english', max_df=0.9,min_df =20)
tfidf = v.fit(Training['tweet'])
#transform trainning&testing data
tfidf = v.transform(Training['tweet'])
tfidfTesting=v.transform(Testing['tweet'])


print('The shape of tf-idf sparce metrix for trainning data =',tfidf.shape)
print('The shape of tf-idf sparce metrix for testing data =', tfidfTesting.shape)


# In[7]:


# generate the x_train and y_train x_test y_test
x_train = tfidf
y_train = Training.iloc[:,7].values

x_test = tfidfTesting
y_test = Testing.iloc[:,7].values


# In[8]:


# Convert the sparce metrix into data frame for future use
tfidfDF=pd.DataFrame(tfidf.todense())
tfidfTestingDF=pd.DataFrame(tfidfTesting.todense())

tfidfDF.columns = v.get_feature_names()
tfidfTestingDF.columns=v.get_feature_names()

tfidfTestingDF.head()


# ## 1. Multiple layer Percepton (neural network)  Model
# Black-box Modeling  
# 
# The fitted model is used both for direct prediction and further variable selection. (Term reduction). 
# 
#   
# 

# #### 1.1 MLP-NN Model with all 1631 variables(Terms)
# 

# In[9]:


X_train_NN, y_train_NN = tfidfDF.iloc[:, :].values, Training['q_Lag_PriceChange']
X_test_NN, y_test_NN = tfidfTestingDF.iloc[:, :].values, Testing['q_Lag_PriceChange']
feature_names = tfidfDF.columns[:].values


# In[10]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(max_iter=1000, random_state=0,hidden_layer_sizes=(200,))
clf.fit(X_train_NN, y_train_NN)


# In[11]:


y_pred_train_NN=clf.predict(X_train_NN) 
y_pred_test_NN=clf.predict(X_test_NN)

Training['MLP-NN Predict'] = pd.Series(y_pred_train_NN)
Testing['MLP-NN Predict'] = pd.Series(y_pred_test_NN)


accuracy_train_NN = accuracy_score(y_train, y_pred_train_NN)
accuracy_test_NN = accuracy_score(y_test, y_pred_test_NN)


print('Accuracy on predict Price change for NN Model on the training set =', np.round(accuracy_train_NN,4))
print('Accuracy on predict Price change for NN Model on the testing set =', np.round(accuracy_test_NN,4))


# ### Variable Importance (VI)
# - One can evaluate the variable importance through the metric score variation caused by the feature value permutation. Ideally, if one feature (variable) is important to the model, a small value perturbation leads to a significant change on the metric score such as the accuracy and $R^2$. 
# - We use the package `eli5` ([github](https://github.com/TeamHG-Memex/eli5), [API doc](https://eli5.readthedocs.io/en/latest/overview.html)) to perform the permutation based importance evaluation.

# In[12]:


# define a permutation importance object
# a gerneral way to calculate importance
# some machine learning method like random forest has build-in ways to calculate Variable Importance
perm = PermutationImportance(clf).fit(X_train_NN, y_train_NN)
# show the importance
eli5.show_weights(perm, feature_names=feature_names)


# In[13]:


# importance in decreasing order
imp_ord = np.argsort(perm.feature_importances_)

plt.figure(figsize=(12,20))
yaxis = np.arange(len(perm.feature_importances_))*1.2
plt.barh(y = yaxis,width = perm.feature_importances_[imp_ord])
plt.yticks(yaxis,feature_names[imp_ord])
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.show()


# ## Select the top 100 important terms to run the model

# In[14]:


WeightDF=eli5.explain_weights_df(perm, feature_names=feature_names)


# ![100Words.png](attachment:100Words.png)

# In[15]:


imp_100 = WeightDF["feature"][0:100].tolist()
impX_100_DF=tfidfDF[imp_100].copy()
impX_100_DF_Testing=tfidfTestingDF[imp_100].copy()
impX_100_DF.head()


# #### 1.2 MLP-NN Model with the top 100 most important variables(Terms)
# 
# 

# In[16]:


X_train_NN_imp100, y_train_NN_imp100 = impX_100_DF.iloc[:, :].values, Training['q_Lag_PriceChange']
X_test_NN_imp100, y_test_NN_imp100 = impX_100_DF_Testing.iloc[:, :].values, Testing['q_Lag_PriceChange']
feature_names_imp100 = impX_100_DF.columns[:].values

clf_imp100 = MLPClassifier(max_iter=1000, random_state=0,hidden_layer_sizes=(200,))
clf_imp100.fit(X_train_NN_imp100, y_train_NN_imp100)

y_pred_train_NN_imp100=clf_imp100.predict(X_train_NN_imp100) 
y_pred_test_NN_imp100=clf_imp100.predict(X_test_NN_imp100)



Training['MLP-NN-100 Predict'] = pd.Series(y_pred_train_NN_imp100)
Testing['MLP-NN-100 Predict'] = pd.Series(y_pred_test_NN_imp100)



accuracy_train_NN_imp100 = accuracy_score(y_train, y_pred_train_NN_imp100)
accuracy_test_NN_imp100 = accuracy_score(y_test, y_pred_test_NN_imp100)


print('Accuracy on predict Price change for NN Model (imp100ortant variables) on the training set =', np.round(accuracy_train_NN_imp100,4))
print('Accuracy on predict Price change for NN Model (imp100ortant variables) on the testing set =', np.round(accuracy_test_NN_imp100,4))


# ## Select the top 19 important terms shown in the eli5.show_weights table to run the model

# In[17]:


names=['friday','game','dia','trump','tariffic','speech','jobsreport','throws','china','day','facebook','iwm','androsform','thursday','tarrifs','talks','philstockworld','moneyempire','rt']
impX_DF=tfidfDF[names].copy()
impX_DF_Testing=tfidfTestingDF[names].copy()
impX_DF.head()


# #### 1.3 MLP-NN Model with the top 19 most important variables(Terms)
# 
# 
# 

# In[18]:


X_train_NN_imp, y_train_NN_imp = impX_DF.iloc[:, :].values, Training['q_Lag_PriceChange']
X_test_NN_imp, y_test_NN_imp = impX_DF_Testing.iloc[:, :].values, Testing['q_Lag_PriceChange']
feature_names_imp = impX_DF.columns[:].values

clf_Imp = MLPClassifier(max_iter=1000, random_state=0,hidden_layer_sizes=(250,))
clf_Imp.fit(X_train_NN_imp, y_train_NN_imp)

y_pred_train_NN_imp=clf_Imp.predict(X_train_NN_imp) 
y_pred_test_NN_imp=clf_Imp.predict(X_test_NN_imp)



Training['MLP-NN-19 Predict'] = pd.Series(y_pred_train_NN_imp)
Testing['MLP-NN-19 Predict'] = pd.Series(y_pred_test_NN_imp)



accuracy_train_NN_imp = accuracy_score(y_train, y_pred_train_NN_imp)
accuracy_test_NN_imp = accuracy_score(y_test, y_pred_test_NN_imp)


print('Accuracy on predict Price change for NN Model (important variables) on the training set =', np.round(accuracy_train_NN_imp,4))
print('Accuracy on predict Price change for NN Model (important variables) on the testing set =', np.round(accuracy_test_NN_imp,4))


# In[19]:


from sklearn.inspection import plot_partial_dependence as pdp
#plot_partial_dependence is introduced in the latest version of sklearn
n_cols = 4
n_rows = feature_names_imp.shape[0]//n_cols + 1

fig = plt.figure(figsize=(15, 2*n_rows))
#pdp (classification, which data you use to draw,...)
pdp(clf_Imp, impX_DF, features=feature_names_imp, feature_names=feature_names_imp,
    n_cols=n_cols, fig=fig, line_kw={'marker': 'o', 'markeredgecolor': 'None'},target=0)

plt.suptitle('Partial Dependence Plots', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.97])


# ![NN.png](attachment:NN.png)

# ## 2. Naive Bayes Model

# #### 2.1 Naive Bayes Model with all 1631 variables(Terms)

# In[20]:


#Fit MultinomialNB model with trainning data
nb = MultinomialNB()            # MultinomialNP(alpha=0.5)
nb.fit(tfidf, Training.q_Lag_PriceChange) 

#Predict Training data
y_pred_train_NB=nb.predict(tfidf) 
Training['NB Predict'] = pd.Series(nb.predict(tfidf))

#Predict Testing data
y_pred_test_NB=nb.predict(v.transform(Testing['tweet']))
Testing['NB Predict'] = pd.Series(nb.predict(tfidfTesting))


Actural_training_NB = Training.iloc[:,7].values
Predict_training_NB = y_pred_train_NB
accuracy_Price_Change_training_NB = accuracy_score(Actural_training_NB, Predict_training_NB)


Actural_Testing_NB = Testing.iloc[:,7].values
Predict_Testing_NB = y_pred_test_NB
accuracy_Price_Change_testing_NB = accuracy_score(Actural_Testing_NB, Predict_Testing_NB)


print('Accuracy on predict Price change for NB Model on trainning data =', np.round(accuracy_Price_Change_training_NB,4))
print('Accuracy on predict Price change for NB Model on testing data =', np.round(accuracy_Price_Change_testing_NB,4))


# #### 2. 2 Naive Bayes Model with the top 100 most important variables(Terms)

# In[29]:


#Fit MultinomialNB_imp100 model with trainning data
NB_imp100 = MultinomialNB()            # MultinomialNP(alpha=0.5)
NB_imp100.fit(impX_100_DF.iloc[:, :].values, Training.q_Lag_PriceChange) 


#Predict Training data
NB_imp100.predict(impX_100_DF.iloc[:, :].values) 
Training['NB-100 Predict'] = pd.Series(NB_imp100.predict(impX_100_DF.iloc[:, :].values))

#Predict Testing data
NB_imp100.predict(impX_100_DF_Testing.iloc[:, :].values)
Testing['NB-100 Predict'] = pd.Series(NB_imp100.predict(impX_100_DF_Testing.iloc[:, :].values))

Actural_training_NB_imp100 = Training.iloc[:,7].values
Predict_training_NB_imp100 = NB_imp100.predict(impX_100_DF.iloc[:, :].values) 
accuracy_Price_Change_training_NB_imp100 = accuracy_score(Actural_training_NB_imp100, Predict_training_NB_imp100)




Actural_Testing_NB_imp100 = Testing.iloc[:,7].values
Predict_Testing_NB_imp100 = NB_imp100.predict(impX_100_DF_Testing.iloc[:, :].values)
accuracy_Price_Change_testing_NB_imp100 = accuracy_score(Actural_Testing_NB_imp100, Predict_Testing_NB_imp100)


print('Accuracy on predict Price change for NB_imp(100 variables) Model on trainning data =', np.round(accuracy_Price_Change_training_NB_imp100,4))
print('Accuracy on predict Price change for NB_imp(100 variables) Model on testing data =', np.round(accuracy_Price_Change_testing_NB_imp100,4))


# #### 2. 3 Naive Bayes Model with the top 19 most important variables(Terms)

# In[28]:


#Fit MultinomialNB_imp model with trainning data
NB_imp = MultinomialNB()            # MultinomialNP(alpha=0.5)
NB_imp.fit(impX_DF.iloc[:, :].values, Training.q_Lag_PriceChange) 


#Predict Training data
NB_imp.predict(impX_DF.iloc[:, :].values) 
Training['NB-19 Predict'] = pd.Series(NB_imp.predict(impX_DF.iloc[:, :].values))

#Predict Testing data
NB_imp.predict(impX_DF_Testing.iloc[:, :].values)
Testing['NB-19 Predict'] = pd.Series(NB_imp.predict(impX_DF_Testing.iloc[:, :].values))

Actural_training_NB_imp = Training.iloc[:,7].values
Predict_training_NB_imp = NB_imp.predict(impX_DF.iloc[:, :].values)
accuracy_Price_Change_training_NB_imp = accuracy_score(Actural_training_NB_imp, Predict_training_NB_imp)




Actural_Testing_NB_imp = Testing.iloc[:,7].values
Predict_Testing_NB_imp = NB_imp.predict(impX_DF_Testing.iloc[:, :].values)
accuracy_Price_Change_testing_NB_imp = accuracy_score(Actural_Testing_NB_imp, Predict_Testing_NB_imp)


print('Accuracy on predict Price change for NB_imp(19 variables) Model on trainning data =', np.round(accuracy_Price_Change_training_NB_imp,4))
print('Accuracy on predict Price change for NB_imp(19 variables) Model on testing data =', np.round(accuracy_Price_Change_testing_NB_imp,4))


# ![NB.png](attachment:NB.png)

# ## 3. Logistic model

# #### 3.1 Logistic Model with all 1631 variables(Terms)

# In[23]:


#Fit a logistic model on the training data with selected features
#by package sklearn LogisticRegression

logreg = LogisticRegression(C=1e8, solver='newton-cg')
logreg.fit(x_train, y_train)

y_pred_train_Logistic = logreg.predict(x_train)
y_pred_test_Logistic = logreg.predict(x_test)

Training['Logistic Predict'] = y_pred_train_Logistic
Testing['Logistic Predict'] = y_pred_test_Logistic

# The fitted model has the coefficients of:
#print("Coefficients :", np.round(logreg.intercept_,4), np.round(logreg.coef_,4))

accuracy_train_Logistic = accuracy_score(y_train, y_pred_train_Logistic)
accuracy_test_Logistic = accuracy_score(y_test, y_pred_test_Logistic)


print('Accuracy on predict Price change for Logistic Model on the training set =', np.round(accuracy_train_Logistic,4))
print('Accuracy on predict Price change for Logistic Model on the testing set =', np.round(accuracy_test_Logistic,4))



# #### 3.2 Logistic Model with the top 100 most important variables(Terms)

# In[24]:


#Fit a logistic model on the training data with selected features
#by package sklearn LogisticRegression

logreg = LogisticRegression(C=1e8, solver='newton-cg')
logreg.fit(impX_100_DF, y_train)

y_pred_train_Logistic_100 = logreg.predict(impX_100_DF)
y_pred_test_Logistic_100 = logreg.predict(impX_100_DF_Testing)

Training['Logistic-100 Predict'] = y_pred_train_Logistic_100
Testing['Logistic-100 Predict'] = y_pred_test_Logistic_100

# The fitted model has the coefficients of:
#print("Coefficients :", np.round(logreg.intercept_,4), np.round(logreg.coef_,4))

accuracy_train_Logistic = accuracy_score(y_train, y_pred_train_Logistic_100)
accuracy_test_Logistic = accuracy_score(y_test, y_pred_test_Logistic_100)


print('Accuracy on predict Price change for Logistic Model(100 variable) on the training set =', np.round(accuracy_train_Logistic,4))
print('Accuracy on predict Price change for Logistic Model(100 variable) on the testing set =', np.round(accuracy_test_Logistic,4))




# #### 3.3 Logistic Model with the top 19 most important variables(Terms)

# In[27]:


#Fit a logistic model on the training data with selected features
#by package sklearn LogisticRegression

logreg = LogisticRegression(C=1e8, solver='newton-cg')
logreg.fit(impX_DF, y_train)

y_pred_train_Logistic = logreg.predict(impX_DF)
y_pred_test_Logistic = logreg.predict(impX_DF_Testing)

Training['Logistic-19 Predict'] = y_pred_train_Logistic
Testing['Logistic-19 Predict'] = y_pred_test_Logistic

# The fitted model has the coefficients of:
print("Coefficients :", np.round(logreg.intercept_,4), np.round(logreg.coef_,4))
accuracy_train_Logistic = accuracy_score(y_train, y_pred_train_Logistic)
accuracy_test_Logistic = accuracy_score(y_test, y_pred_test_Logistic)


print('Accuracy on predict Price change for Logistic Model(19 variable) on the training set =', np.round(accuracy_train_Logistic,4))
print('Accuracy on predict Price change for Logistic Model(19 variable) on the testing set =', np.round(accuracy_test_Logistic,4))





# ![LOG.png](attachment:LOG.png)

# ## Model Evaluation
# 
# 
# ![Evaluation.jpeg](attachment:Evaluation.jpeg)

# In[26]:


Training.to_csv(r'/Users/zhangxinyi/Desktop/Tf-Idf_Training.csv')
Testing.to_csv(r'/Users/zhangxinyi/Desktop/Tf-Idf_Testing.csv')
WeightDF.to_csv(r'/Users/zhangxinyi/Desktop/Tf-Idf_ImportanceWordList.csv')

