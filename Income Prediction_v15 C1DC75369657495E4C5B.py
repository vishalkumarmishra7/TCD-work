#!/usr/bin/env python
# coding: utf-8

# #### Importing libraries

# In[1]:


# from keras.callbacks import ModelCheckpoint
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error 
# from matplotlib import pyplot as plt
# import seaborn as sb
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import warnings 
# warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# from xgboost import XGBRegressor


# In[2]:


import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import joblib
# from sklearn.metrics import accuracy_score
import category_encoders as ce
from catboost import CatBoostRegressor
import random
np.random.seed(23)


# #### Reading data set

# In[3]:


def train_test():
    df = pd.read_csv('../Data/tcd ml 2019-20 income prediction training (with labels).csv')
    df['Income in EUR2'] = pd.qcut(df['Income in EUR'], 10, labels=False)
    y = df.pop('Income in EUR2')
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.15, stratify=y)
    df_val = X_test
    df = X_train
    return df_val, df


# #### Outlier Detection

# In[4]:


def rem_outliers(df):
    df_ = df.copy()
    median = df['Income in EUR'].median()
    std = df['Income in EUR'].std()
    df_ = df_[df_['Income in EUR'] >= median - (2*std)]
    df_ = df_[df_['Income in EUR'] <=  median + (2*std)]
    return df


# #### Creating Train Test

# In[5]:


def freq_chart_cat(df,a):
    df2 = df.groupby(a).agg({'Instance':'count'}).sort_values(by=a)
    ax = df2['Instance'].plot(kind='bar', title =a+" Freq", figsize = [df2.shape[0]*0.5,5], legend=True, fontsize=18)
    ax.set_xlabel(a, fontsize=20)
    ax.set_ylabel("Freq", fontsize=20)
    print(a+' Frequency check \n')
    print(df2)
    print(plt.show())
    print('\n\n')

def num_hist(df,a):
    plt.hist(df[a], normed=True, bins=20)
    plt.ylabel('Probability')
    plt.xlabel(a)
    print(a+' Frequency check \n')
    print(plt.show())
    print('\n\n')

def data_explore(df, charts = 'False'):
    unique_size = []
    zeros = []
    cols = []
    blanks = []
    unknowns = []
    
    for i in df:
        cols.append(i)
        unique_size.append(str(round(df[i].nunique()*100/df.shape[0],2))+'%')

        l = list(df[i])
        n = 0
        m = 0
        u = 0
        for j in l:
            if str(j).strip() == '0':
                n += 1
            if str(j).strip == '':
                m += 1
            if str(j).lower() == 'unknown':
                u += 1
        zeros.append(n)
        blanks.append(m)
        unknowns.append(u)
        
    df_info2 = pd.DataFrame({'Data Type':df.dtypes.tolist(), 'Number of Nulls':df.isna().sum().tolist(), 'Number of Zeroes':zeros, 'Number of Blanks':blanks, 'Number of Unknowns':unknowns, 'Percentage of Unique Values':unique_size})
    df_info2.index = cols
    df_describe = pd.DataFrame(df.describe()).transpose()
    df_info3 = df_info2.merge(df_describe,left_on = df_info2.index, right_on = df_describe.index, how = 'left')
    df_info3 = df_info3.rename(columns = {'key_0':'columns'})
    df_info3 = df_info3.set_index('columns')
    
    if charts == 'True':
        for i in df.columns:
            if df[i].nunique() < 20:
                freq_chart_cat(df,i)
            if df[i].dtype.kind in 'bifc':
                num_hist(df,i)
    return df_info3


# ### Data Cleaning
# 1. Converting to lowercase
# 2. replacing unknowns by NaN
# 3. imputing nulls

# In[6]:


def zero_to_nulls(df):
    for i in df:
        if df[i].dtype.kind in 'bifuc':
            df.loc[:,i] = df.loc[:,i].replace(0,np.nan)
        else:
            df.loc[:,i] = df.loc[:,i].replace('0',np.nan)
    return df

def unknown_to_nulls(df):
    for i in df:
        if df[i].dtype.kind not in 'bifuc':
            df.loc[:,i] = df.loc[:,i].replace('unknown',np.nan)
    return df

def convert_lower(df):
    for i in df:
        if df[i].dtype.kind not in 'biufc':
            df.loc[:,i] = df.loc[:,i].str.lower()
            df.loc[:,i] = df.loc[:,i].str.strip()
    return df


# In[7]:


def cleaner1(df):
    df2 = convert_lower(df)
    df2 = unknown_to_nulls(df2)
    df2.loc[:,'Wears Glasses'] = df2.loc[:,'Wears Glasses'].apply(str)
    df2.loc[:,'Wears Glasses'] = np.where(df2.loc[:,'Wears Glasses'] == '0', 'No',np.where(df2.loc[:,'Wears Glasses'] == '1', 'Yes',df2.loc[:,'Wears Glasses']))
    df2.loc[:,'Gender'] = df2.loc[:,'Gender'].replace('other',np.nan)
    df2 = zero_to_nulls(df2)
    return df2


# #### Treating profession variable

# In[8]:


def unique(list1): 
    # intilize a null list 
    unique_list = []       
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

def profession_cleaner(df, df2, stretch = 0.8):
    dfx = df[df['Profession'].notnull()]

    dfx = dfx[['Profession','Income in EUR']]

    dfx['Profession'] = dfx['Profession'].str.lower()

    chars = []
    for i in list(dfx['Profession']):
        for j in i:
            if (ord(j) in list(range(65,91))) or (ord(j) in list(range(97,123))):
                chars = chars
            else:
                chars.append(j)

    for i in unique(chars):
        if i != ' ':
            dfx['Profession'] = dfx['Profession'].str.replace(i,' ')

    df_prof = dfx['Profession'].str.split(' ', 10, expand=True)

    df_prof['Income in EUR'] = dfx['Income in EUR']

    l = []
    val = []
    for i in df_prof.columns[:-1]:
        df_prof2 = df_prof[df_prof[i].notnull()]
        m = df_prof2.iloc[:,i].tolist()
        n = df_prof2['Income in EUR'].tolist()
        for j in m:
            l.append(j)
        for k in n:
            val.append(k)

    df_prof3 = pd.DataFrame({'Words':l,'Income in EUR':val})
    df_prof4 = df_prof3.groupby('Words',as_index = False).agg({'Income in EUR':['mean','var','count']})
    df_prof4.columns = df_prof4.columns.droplevel(level=0)
    df_prof4['var_sum'] = df_prof4['var'] * df_prof4['count']
    df_prof4 = df_prof4.drop(columns = ['count'])
    df_prof4.columns = ['Words','mean','var','count']

    df_prof4 = df_prof4[df_prof4['Words'] != ' ']
    df_prof4 = df_prof4[df_prof4['Words'] != '']
    df_prof4 = df_prof4[df_prof4['Words'] != 'and']

    df_prof5 = df_prof4.copy()

    df_prof5['total_word_count'] = df_prof5['count'].sum()

    df_prof5 = df_prof5.sort_values(by = 'count', ascending = False)

    df_prof5['cnt_cumsum'] = df_prof5['count'].cumsum()

    df_prof5['cum_perc'] = df_prof5['cnt_cumsum']/df_prof5['total_word_count']

    df_prof5 = df_prof5[df_prof5['cum_perc'] <= stretch]

#     print('Number of Significant words = ')
#     print(df_prof5.shape[0])

    sig_words = list(df_prof5['Words'])
#     print('Significant words are')
#     print(sig_words)

    extra_cols = []
    for i in sig_words:
        df2['is_'+i] = np.where(df2['Profession'].str.contains(i),'yes','no')
        extra_cols.append('is_'+i)
    
    return df2, extra_cols, sig_words


# #### Creating a missing value imputer

# In[9]:


def miss_val_impute(df, cat_vars, num_imp = 'mean'):
    for i in df:
        if i not in cat_vars:
            if num_imp == 'mean':
                x = df[i].mean()
                df[i] = df[i].replace(np.nan,x)
            if num_imp == 'median':
                x = df[i].median()
                df[i] = df[i].replace(np.nan,x)
            if num_imp == 'mode':
                x = df[i].mode()
                df[i] = df[i].replace(np.nan,x)
        else:
            df[i] = np.where(df[i].isna(),df[i].mode(),df[i])
    return df

def mvi(df2, extra_cols):
    df3 = df2.copy()
    cat_vars = ['Gender','Country','Profession','University Degree','Wears Glasses','Hair Color'] + extra_cols
    df3 = miss_val_impute(df3, cat_vars,num_imp = 'median')
    return df3,cat_vars


# #### Label Encoding

# In[10]:


# dfx = df[df['Profession'].notnull()]

# dfx['Profession'] = dfx['Profession'].str.lower()

# df_prof = dfx['Profession'].str.split(' ', 10, expand=True)

# df_prof['Income in EUR'] = dfx['Income in EUR']

# l = []
# val = []
# for i in df_prof.columns[:-1]:
#     df_prof2 = df_prof[df_prof[i].notnull()]
#     m = df_prof2.iloc[:,i].tolist()
#     n = df_prof2['Income in EUR'].tolist()
#     for j in m:
#         l.append(j)
#     for k in n:
#         val.append(k)
    
# df_prof3 = pd.DataFrame({'Words':l,'Income in EUR':val})
# df_prof4 = df_prof3.groupby('Words',as_index = False).agg({'Income in EUR':['mean','var','count']})
# df_prof4.columns = df_prof4.columns.droplevel(level=0)
# # df_prof4['var_sum'] = df_prof4['var'] * df_prof4['count']
# df_prof4 = df_prof4.drop(columns = ['count'])
# df_prof4.columns = ['Words','mean','var']
# print(df_prof4.head())

# d_prof = {}
# for i in dfx['Profession'].unique():
#     least_var = 10**20
#     for j in df_prof4['Words'].tolist():
#         if j in i:
#             a = df_prof4[df_prof4['Words'] == j]['var'].tolist()
#             b = df_prof4[df_prof4['Words'] == j]['mean'].tolist()
#             if a[0] < least_var:
#                 least_var = a[0]
#                 d_prof[i] = b[0]

# df3['Profession2'] = df3['Profession'].map(d_prof)
# cat_vars.append('Profession2')


# In[11]:


def encoder(df3,cat_vars):
    df4 = df3.copy()
    for i in cat_vars:
        mean_encode = df4.groupby(i)['Income in EUR'].mean()
        freq_encode = df4.groupby(i)['Income in EUR'].count()
        df4.loc[:, i] = (df4[i].map(mean_encode)*df4[i].map(freq_encode) + 7*df4['Income in EUR'].mean())/(df4[i].map(freq_encode)+7)
    return df4


# In[12]:


# area = np.pi*3

# for i in df4.drop(columns = ["Income in EUR",'Instance']).columns:
#     x = df4[i]
#     y = df4['Income in EUR']
#     plt.scatter(x, y, s=area, alpha=0.5)
#     plt.xlabel(i)
#     plt.ylabel('Income in EUR')
#     plt.show()


# In[13]:


def train_data_prep(df4):
    df = df4.drop(columns = ['Instance'])
    df['Income in EUR2'] = pd.qcut(df['Income in EUR'], 10, labels=False)
    y = df.pop('Income in EUR2')
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42, stratify=y)
    
    return X_train.drop(columns = ['Income in EUR']), X_train[['Income in EUR']], X_test.drop(columns = ['Income in EUR']), X_test[['Income in EUR']]


# In[14]:


# clf = RandomForestRegressor(n_estimators=100)
# clf = clf.fit(X_train, Y_train)
def modeller(X_train, Y_train, X_test, Y_test, extra_cols):
    X_train.columns = ['Year_of_Record','Gender','Age','Country','Size_of_City','Profession','University_Degree','Wears_Glasses','Hair_Color','Body_Height_cm'] + extra_cols
    X_test.columns = ['Year_of_Record','Gender','Age','Country','Size_of_City','Profession','University_Degree','Wears_Glasses','Hair_Color','Body_Height_cm'] + extra_cols
#     xg_reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#            colsample_bytree=1.0, gamma=0, learning_rate=0.1, max_delta_step=0,
#            max_depth=6, min_child_weight=1.2, missing=None, n_estimators=100,
#            n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#            silent=True, subsample=1)
#     clf = xg_reg.fit(X_train, Y_train)
    
    clf = CatBoostRegressor(iterations=1400,
                             learning_rate=0.02,
                             depth=6,
                             eval_metric='RMSE',
#                              silent = True,
                             random_seed = 23,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 250,
                             od_wait=100)
    clf.fit(X_train, Y_train,
                 eval_set=(X_test,Y_test),
#                  cat_features=cat_vars_l,
                 use_best_model=True,
                 verbose=True)
    return clf


# In[15]:


# X_test.columns = ['Year_of_Record','Gender','Age','Country','Size_of_City','Profession','University_Degree','Wears_Glasses','Hair_Color','Body_Height_cm'] + extra_cols
# print(clf.score(X_test,Y_test))
# rms = sqrt(mean_squared_error(Y_test, clf.predict(X_test)))
# rms


# In[16]:


# importances = list(clf.feature_importances_)
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X.columns, importances)]
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[17]:


# X.columns = ['Year_of_Record','Gender','Age','Country','Size_of_City','Profession','University_Degree','Wears_Glasses','Hair_Color','Body_Height_cm'] + extra_cols
# scores = cross_val_score(clf, X, Y, cv=5)
# print(scores.mean())
# scores


# In[18]:


def result(clf,extra_cols,sig_words,df3,mode='check'):
    if mode == 'check':
        dfkaggle = df_val
    else:
        dfkaggle = pd.read_csv('../Data/tcd ml 2019-20 income prediction test (without labels).csv')

    dfkaggle2 = convert_lower(dfkaggle)
    dfkaggle2 = unknown_to_nulls(dfkaggle2)
    dfkaggle2.loc[:,'Wears Glasses'] = dfkaggle2.loc[:,'Wears Glasses'].apply(str)
    dfkaggle2.loc[:,'Wears Glasses'] = np.where(dfkaggle2.loc[:,'Wears Glasses'] == '0', 'No',np.where(dfkaggle2.loc[:,'Wears Glasses'] == '1', 'Yes',dfkaggle2.loc[:,'Wears Glasses']))
    dfkaggle2.loc[:,'Gender'] = dfkaggle2.loc[:,'Gender'].replace('other',np.nan)
    dfkaggle2 = zero_to_nulls(dfkaggle2)

    for i in sig_words:
        dfkaggle2['is_'+i] = np.where(dfkaggle2['Profession'].str.contains(i),'yes','no')

    def miss_val_impute2(df, cat_vars, num_imp = 'mean'):
        for i in df:
            if i not in cat_vars:
                if num_imp == 'mean':
                    x = df3[i].mean()
                    df[i] = df[i].replace(np.nan,x)
                if num_imp == 'median':
                    x = df3[i].median()
                    df[i] = df[i].replace(np.nan,x)
                if num_imp == 'mode':
                    x = df3[i].mode()
                    df[i] = df[i].replace(np.nan,x)
            else:
                df[i] = np.where(df[i].isna(),df3[i].mode(),df[i])
        return df

    dfkaggle3 = dfkaggle2.copy()
    cat_vars = ['Gender','Country','Profession','University Degree','Wears Glasses','Hair Color'] + extra_cols
    
    if mode == 'check':
        dfkaggle3 = dfkaggle3.drop(columns = 'Income in EUR')
    else:
        dfkaggle3 = dfkaggle3.drop(columns = 'Income')
        
    dfkaggle3 = miss_val_impute2(dfkaggle3, cat_vars,num_imp = 'median')
    dfkaggle3.head()

    dfkaggle4 = dfkaggle3.copy()
    for i in cat_vars:
        mean_encode = df3.groupby(i)['Income in EUR'].mean()
#         dfkaggle4.loc[:, i] = dfkaggle4[i].map(mean_encode)
        freq_encode = df3.groupby(i)['Income in EUR'].count()
        dfkaggle4.loc[:, i] = (dfkaggle4[i].map(mean_encode)*dfkaggle4[i].map(freq_encode) + 7*df3['Income in EUR'].mean())/(dfkaggle4[i].map(freq_encode)+7)
    dfkaggle4.head()

    data_explore(dfkaggle4)
    dfkaggle4 = dfkaggle4.drop(columns = 'Instance')

    for i in dfkaggle4:
        dfkaggle4[i] = np.where(dfkaggle4[i].isna(),df4[i].median(),dfkaggle4[i])

    # dfkaggle4 = dfkaggle4.fillna(0)

    dfkaggle4.columns = ['Year_of_Record','Gender','Age','Country','Size_of_City','Profession','University_Degree','Wears_Glasses','Hair_Color','Body_Height_cm'] + extra_cols

    dfkaggle4['Income in EUR'] = clf.predict(dfkaggle4)
    dfkaggle4['Instance'] = dfkaggle3['Instance']
    dfkaggle4 = dfkaggle4[['Instance','Income in EUR']]
#     dfkaggle4.columns = ['Instance','Income in EUR']
    
    if mode == 'check':
        rms = sqrt(mean_squared_error(list(df_val['Income in EUR']), list(dfkaggle4['Income in EUR'])))
        return rms
    else:
        return dfkaggle4


# In[19]:


# # ### Splitting dataframe
scores = []
output = pd.DataFrame()
iters = []
for i in range(15):
    df_val, df = train_test()
    df = rem_outliers(df)
    df2 = cleaner1(df)
    df2, extra_cols, sig_words = profession_cleaner(df, df2, stretch = 0.2)
    df3,cat_vars = mvi(df2, extra_cols)
    df4 = encoder(df3,cat_vars)
    X_train, Y_train, X_test, Y_test = train_data_prep(df4)
    clf = modeller(X_train, Y_train, X_test, Y_test, extra_cols)
    
    rms = result(clf,extra_cols,sig_words,df3,mode  = 'check')
    print('iteration = '+str(i)+' score = '+str(rms))
    joblib.dump(clf, 'Models for v10 KFolds/clf'+str(i))
    
    if rms <= 60000:
        scores.append(rms)
        print('iteration = '+str(i)+' score = '+str(rms))
        res = result(clf,extra_cols,sig_words,df3,mode = 'kaggle')
        output = output.append(res)
        print('Completed iteration : ' + str(i))
        iters.append(i)
    
print('Mean score = ' + str(sum(scores)/len(scores)))


# In[22]:


# output2 = output.copy()
# # # output2
# output3 = output2.groupby(['Instance']).mean()
# output3.head()
# output3.to_csv('test15 catboost.csv')

