#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
dest = 'Step1 - Permutation Test/'
# import shap
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')


# ### Reading and filtering reviews dataset

# In[2]:


df = pd.read_csv(dest+"Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
df2 = df[['primaryCategories','reviews.rating','reviews.text','reviews.title','reviews.date']]
df2.head()


# ### Cleaning text
# - Removing punctuations and stop words
# - Experimentally removing words containing numbers
# - Removing non-english words

# In[3]:


stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())
l = []
for i in list(df2['reviews.text']):
    example_sent = i
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(example_sent)
    filtered_sentence = [w for w in word_tokens if not w in stop_words and w in words and any(char.isdigit() for char in w) == False]
    l.append(' '.join(filtered_sentence))
    
df2.loc[:,'reviews.text'] = np.array(l)

# not w in stop_words and


# ### Creating features

# In[4]:


vectorizer = CountVectorizer(analyzer='word',ngram_range=(1, 1))
X = vectorizer.fit_transform(df2['reviews.text'])
df3 = pd.DataFrame(X.toarray())
df3.columns = vectorizer.get_feature_names()
df3.head()


# ### Creating flag which is 1 if the date of review is on a weekday and 0 if it is on a weekend

# In[5]:


# df3['TimeCycle'] = np.where(pd.to_datetime(df2['reviews.date'].str[:10], format='%Y-%m-%d').dt.dayofweek < 5,1,0)
# df3['TimeCycle'] =  pd.to_datetime(df2['reviews.date'].str[:10], format='%Y-%m-%d').dt.day
df3['TimeCycle'] = df2['reviews.rating']
df3.head()


# ### Train test validation split

# In[6]:


s = np.random.uniform(low = 0, high = df3.shape[0] - 1, size = int(0.15*df3.shape[0]))
s = np.unique(s.round(0))

dfv = df3.iloc[s,:]
df4 = df3.drop(s)

X_train, X_test, y_train, y_test = train_test_split(df4.drop(columns = ['TimeCycle']), df4['TimeCycle'], test_size=0.30, random_state=1)


# ### Model Training

# In[7]:


train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_test, label=y_test.values)

model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.1,
    random_strength=0.1,
    depth=6,
    metric_period = 1000,
    eval_metric='Accuracy',
    task_type = "GPU",
    devices = '0:1'
)

model.fit(train_pool,plot=True,eval_set=test_pool)


# ### Making Prediction and calculating accuracy

# In[8]:


result = model.predict(dfv.drop(columns = 'TimeCycle')).round(0)
final_accuracy = acc(dfv['TimeCycle'],result)*100


# ### Permutation Test for significance of accuracy through model
# 
# - The red line presents the accuracy achieved though the model
# - The blue histogram shows the accuracy if we randomly chose the label using a uniform distribution between minimum and maximum values of the label (day of week, day of month etc.)
# - The green histogram shows the accuracy of the model if if simply shuffled the labels of test data
# 
# #### Conclusion:
# Since the accuracy from the model is above 99th percent of random combinations, we prove that the text is able to predict the time cycle i.e. the label.
# 
# Note: Here we have predicted weather the day of the review is a weekend or not. Similarly we can do it for day of month.

# In[9]:


acc_set = [];
acc_set2 = [];

np_y_test = np.random.uniform(low = min(dfv['TimeCycle']), high = max(dfv['TimeCycle']), size = len(dfv['TimeCycle'])).round(0)
np_y_test2 = np.array(dfv['TimeCycle'])

for i in range(10000):
    np.random.shuffle(np_y_test)
    np.random.shuffle(np_y_test2)
    acc_set.append(acc(np_y_test,result)*100)
    acc_set2.append(acc(np_y_test2,result)*100)


# ### Comparing model accuracy from histogram of accuracies from permutations of uniformy distributed labels

# In[19]:


fig = plt.figure()
plt.hist(acc_set, bins=50, color = 'b')
plt.axvline(x=final_accuracy,color = 'r')
plt.savefig(dest+'fig1.jpg')
plt.show()
fig.savefig(dest+'fig1.png', dpi=fig.dpi)
print("The accuracy from the model is ",scipy.stats.percentileofscore(acc_set, final_accuracy, kind='rank')," percentile in the permutation test using uniform distribution between minimum and maximum values of test data labels")


# ### Comparing model accuracy from histogram of accuracies from permutations of shuffled test data

# In[20]:


fig = plt.figure()
plt.hist(acc_set2, bins=50, color = 'g')
plt.axvline(x=final_accuracy,color = 'r')
plt.show()
fig.savefig(dest+'fig2.png', dpi=fig.dpi)
print("The accuracy from the model is ",scipy.stats.percentileofscore(acc_set2, final_accuracy, kind='rank')," percentile in the permutation test using shuffled values of test data labels")


# ### AUC and Confusion Matrix

# In[22]:


fig = plt.figure()
y_pred_proba = model.predict_proba(dfv.drop(columns = 'TimeCycle'))[::,1]
fpr, tpr, _ = metrics.roc_curve(dfv['TimeCycle'],y_pred_proba)
auc = metrics.roc_auc_score(dfv['TimeCycle'], y_pred_proba)
plt.plot(fpr,tpr,label="ROC, auc="+str(auc))
plt.legend(loc=4)
plt.show()
fig.savefig(dest+'fig3.png', dpi=fig.dpi)

cm = metrics.confusion_matrix(dfv['TimeCycle'],result)
# labels = ['No Default', 'Default']
fig = plt.figure(figsize=(8,6))
sns.heatmap(cm, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()
fig.savefig(dest+'fig4.png', dpi=fig.dpi)


# In[13]:


# shap_values = model.get_feature_importance(Pool(X_test, label=y_test),type="ShapValues")
# expected_value = shap_values[0,-1]
# shap_values = shap_values[:,:-1]
# shap.initjs()
# shap.force_plot(expected_value, shap_values[3,:], X_test.iloc[3,:])


# In[14]:


df5 = pd.DataFrame(data = {'Features': df4.drop(columns = 'TimeCycle').columns, 'Importances':model.get_feature_importance()}).sort_values(by = 'Importances', ascending = False)
df5.to_csv(dest+'Importances.csv')

