#!/usr/bin/env python
# coding: utf-8

# ## import Libraries

# In[37]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgbm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier as dt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import shap
sns.set_style("whitegrid")
# np.random.seed(100)
import datetime
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import random
import category_encoders as ce
# from imblearn.over_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import math
import statistics
import warnings
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings("ignore")


# ## Jabref Cleaner Function

# In[70]:


lo={'recommendation_set_id':str, 'user_id':str, 'session_id':str, 'query_identifier':str,
    'query_word_count':float, 'query_char_count':float, 'query_detected_language':str,
    'query_document_id':str, 'document_language_provided':str, 'year_published':float,
    'number_of_authors':float, 'abstract_word_count':float, 'abstract_char_count':float,
    'abstract_detected_language':str, 'first_author_id':str,
    'num_pubs_by_first_author':float, 'organization_id':str, 'application_type':str,
    'item_type':str, 'request_received':str, 'hour_request_received':str,
    'response_delivered':str, 'rec_processing_time':float, 'app_version':str, 'app_lang':str,
    'user_os':str, 'user_os_version':str, 'user_java_version':str, 'user_timezone':str,
    'country_by_ip':str, 'timezone_by_ip':str, 'local_time_of_request':str,
    'local_hour_of_request':str, 'number_of_recs_in_set':float,
    'recommendation_algorithm_id_used':str, 'algorithm_class':str, 'cbf_parser':str,
    'search_title':str, 'search_keywords':str, 'search_abstract':str,
    'time_recs_recieved':str, 'time_recs_displayed':str, 'time_recs_viewed':str,
    'clicks':float, 'ctr':float,'set_clicked':float
}

# pars=['request_received', 'response_delivered','local_time_of_request','time_recs_recieved','time_recs_displayed','time_recs_viewed']

# df_w=pd.read_csv('../Data/tcdml1920-rec-click-pred--training.csv',na_values=["\\N","nA"],dtype=lo, parse_dates=pars)
# df_kag=pd.read_csv('../Data/tcdml1920-rec-click-pred--test.csv',na_values=["\\N","nA"],dtype=lo, parse_dates=pars)

def process(df,run,dict_key):
    
    df=df[df.organization_id=='1']
    df=df[['query_word_count','query_char_count', 'query_detected_language', 'query_document_id','year_published',
           'number_of_authors','abstract_word_count', 'abstract_char_count','first_author_id','num_pubs_by_first_author',
           'request_received','hour_request_received','rec_processing_time','app_version', 'app_lang','user_os','user_timezone','country_by_ip',
           'timezone_by_ip','local_hour_of_request','recommendation_algorithm_id_used','set_clicked']]

    df['query_detected_language'].fillna(df['query_detected_language'].mode()[0], inplace=True)

    df['query_doc_id_present']=df.query_document_id.isna()*1
    df.drop(columns='query_document_id',inplace=True)

    df.drop(df[df.year_published >2019].index, inplace=True)

    df['day_of_week'] = df['request_received'].dt.day_name()
    df['month'] = df['request_received'].dt.month
    df['year']=df['request_received'].dt.year
    df['month']=df.month.astype(str)
    df.drop(columns=['request_received'],inplace=True)

    if run=='train':
        df=df[df.rec_processing_time<25]
    
    df.drop(columns=['rec_processing_time'],inplace=True)

    df['app_version'].fillna(df['app_version'].mode()[0], inplace=True)

    df['app_lang'].fillna(df['app_lang'].mode()[0], inplace=True)

    df['user_os_Windows_8_1']=df.user_os.map(lambda x: 1 if x=='Windows 8.1' else 0)
    df['user_os_provided']=df.user_os.map(lambda x: 1 if x else 0)
    df.drop(columns='user_os',inplace=True)

    df['user_timezone_present']=df.user_timezone.map(lambda x:1 if x else 0)
    df['user_timezone_aus']=df.user_timezone.map(lambda x:1 if x=='Australia/Sydney' else 0)
    df.drop(columns='user_timezone',inplace=True)

    df.local_hour_of_request.fillna(df.local_hour_of_request.mode()[0],inplace=True) 
    
    df['recommendation_algorithm_id_used'].fillna(df['recommendation_algorithm_id_used'].mode()[0], inplace=True)

#     df['cbf_standard_QP']=df.cbf_parser.map(lambda x:1 if x=='standard_QP' else 0)
#     df['cbf_edismax_QP']=df.cbf_parser.map(lambda x:1 if x=='cbf_edismax_QP' else 0)
#     df['cbf_mlt_QP']=df.cbf_parser.map(lambda x:1 if x=='cbf_mlt_QP' else 0)
#     df['cbf_parser_used']=df.cbf_parser.map(lambda x: 1 if x else 0)
#     df.drop(columns=['cbf_parser'],inplace=True)

    def convert_sparse_values(df, cols, threshold, replacement='other'):
        for col in [cols]:
            counts = df[col].value_counts()
            to_convert = counts[counts <= threshold].index.values
            dict_key[cols]=to_convert
            df[col] = df[col].replace(to_convert, replacement)

                
    if run=='train':
        convert_sparse_values(df,cols='query_detected_language', threshold=1000)
        convert_sparse_values(df,cols='app_lang', threshold=500)
        convert_sparse_values(df,cols='country_by_ip', threshold=150)
        convert_sparse_values(df,cols='timezone_by_ip', threshold=500)
        convert_sparse_values(df,cols='app_version', threshold=800)

    else:
        df['query_detected_language']=df.query_detected_language.map(lambda x: x if x in dict_key['query_detected_language'] else 'others')
        df['app_lang']=df.app_lang.map(lambda x: x if x in dict_key['app_lang'] else 'others')
        df['country_by_ip']=df.country_by_ip.map(lambda x: x if x in dict_key['country_by_ip'] else 'others')
        df['timezone_by_ip']=df.timezone_by_ip.map(lambda x: x if x in dict_key['country_by_ip'] else 'others')
        df['app_version']=df.app_version.map(lambda x: x if x in dict_key['app_version'] else 'others')
        
    df['country_by_ip'].fillna(df['country_by_ip'].mode()[0], inplace=True)
    df.timezone_by_ip.fillna(df.timezone_by_ip.mode()[0],inplace=True) 

    df=df.drop(columns=['set_clicked']).merge(df[['set_clicked']], 
                                    on=df[['set_clicked']].index).drop(columns='key_0')
    
    return df,dict_key


# ## Encoder Function

# In[71]:


def encode_all(df,dfv,dfk,encoder_to_use,handle_missing='return_nan'):
    
    encoders_used = {}
    
    for col in encoder_to_use:

        if encoder_to_use[col] == 'ColumnDropper':
            df = df.drop(columns = col)
            dfv = dfv.drop(columns = col)
            dfk = dfk.drop(columns = col)
            encoders_used[col] = 'ColumnDropper'    
                
        if encoder_to_use[col]=='BackwardDifferenceEncoder':
            encoder=ce.BackwardDifferenceEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='BaseNEncoder':
            encoder=ce.BaseNEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,base=3) 
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='BinaryEncoder':
            encoder=ce.BinaryEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='CatBoostEncoder':
            encoder=ce.CatBoostEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,sigma=None,a=2)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

    #     if encoder_to_use[col]=='HashingEncoder':
    #         encoder=ce.HashingEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
    #         encoder.fit(X=df,y=df['set_clicked'])
    #         df=encoder.transform(df)
    #         encoders_used[col]=encoder

        if encoder_to_use[col]=='HelmertEncoder':
            encoder=ce.HelmertEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='JamesSteinEncoder':
            encoder=ce.JamesSteinEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing, model='binary')
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='LeaveOneOutEncoder':
            encoder=ce.LeaveOneOutEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,sigma=None)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='MEstimateEncoder':
            encoder=ce.MEstimateEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,randomized=True,sigma=None,m=2)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='OneHotEncoder':
            encoder=ce.OneHotEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,use_cat_names=True)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='OrdinalEncoder':
            encoder=ce.OrdinalEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='SumEncoder':
            encoder=ce.SumEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='PolynomialEncoder':
            encoder=ce.PolynomialEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='TargetEncoder':
            encoder=ce.TargetEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,min_samples_leaf=10, smoothing=5)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder


        if encoder_to_use[col]=='WOEEncoder':
            encoder=ce.WOEEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,randomized=True,sigma=None)
            encoder.fit(X=df,y=df['set_clicked'])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder
            
#         print("Encoding done for - ",col)
    
    print("Completed encoder - ",datetime.datetime.now())
    
    return df, dfv, dfk, encoders_used


# ## Imputer Function

# In[72]:


def imputer(df, dfv, dfk, imputer_dict):
    
    result = {}
    
    for i in imputer_dict:
        
        if imputer_dict[i]['Indicator'] == 'deleterows':
            if df[i].isna().sum() > 0:
                df = df[df[i].isfinite()]
                dfv = dfv[dfv[i].isfinite()]
                dfk = dfk[dfk[i].isfinite()]
            
        if imputer_dict[i]['Indicator'] == True:
            if df[i].isna().sum() > 0:
                df[i+'_null_ind'] = np.where(df[i].isna(),1,0)
                dfv[i+'_null_ind'] = np.where(dfv[i].isna(),1,0)
                dfk[i+'_null_ind'] = np.where(dfk[i].isna(),1,0)
        
        if imputer_dict[i]['mvi'] in ['mean','median','most_frequent']:
            imp = SimpleImputer(missing_values = np.nan
                                , strategy = imputer_dict[i]['mvi']
                                , verbose = True
                                , add_indicator = False
                                , fill_value = None
                               )
            imp.fit(df[[i]])
            result[i] = imp
            df.loc[:,i] = result[i].transform(df[[i]])
            dfv.loc[:,i] = result[i].transform(dfv[[i]])
            dfk.loc[:,i] = result[i].transform(dfk[[i]])
        
        if imputer_dict[i]['mvi'] == 'far_val':
            result[i] = df[i].max()*100
            df[i] = np.where(df[i].isna(),result[i],df[i])
            dfv[i] = np.where(dfv[i].isna(),result[i],dfv[i])
            dfk[i] = np.where(dfk[i].isna(),result[i],dfk[i])
        
#         print("Completed imputing : ", i)
        
    ##### interativeimputer (if none of the above then this) ######
    
    imp = IterativeImputer(
                        max_iter = 3
#                        random_state = 0
#                        , add_indicator = False
                       , estimator = ExtraTreesRegressor()
#                        , estimator = None #### default is bayesianridge(), KNeighborsRegressor(n_neighbors = 10)
#                        , imputation_order = 'ascending' #### descending, roman, arabic, random
#                        , initial_strategy = 'mean' #### mean, median, mode
#                        , max_value = None
#                        , min_value = None
#                        , missing_values = np.nan
                       , n_nearest_features = 5 ##### Change value for maximum columns considered to predict missing value
#                        , sample_posterior = False
#                        , tol = 0.001
#                        , verbose = 1
                      )
    
#     dfc = df.copy()
#     df = df.drop(columns = 'set_clicked')
    
#     dfvc = dfv.copy()
#     dfv = dfv.drop(columns = 'set_clicked')

    dfvc = dfv.copy()
    dfv['set_clicked'] = np.nan
    
#     dfkc = dfk.copy()
#     dfk = dfk.drop(columns = 'set_clicked')
    
    dfkc = dfk.copy()
    dfk['set_clicked'] = np.nan
    
    dfcolumns = df.columns
    imp.fit(df)
    df = pd.DataFrame(imp.transform(df))
    df.columns = dfcolumns
    dfv = pd.DataFrame(imp.transform(dfv))
    dfv.columns = dfcolumns
    dfk = pd.DataFrame(imp.transform(dfk))
    dfk.columns = dfcolumns
    
#     df['set_clicked'] = np.array(dfc['set_clicked'])
    dfv['set_clicked'] = np.array(dfvc['set_clicked'].astype('int64'))
    dfk['set_clicked'] = np.nan
    
    for i in imputer_dict:
        if imputer_dict[i]['mvi'] == 'iterativeimputer':
            result[i] = imp
    
    print("Completed imputer - ",datetime.datetime.now())
    
    return df, dfv, dfk, result


def read_data(org):
    
    df = pd.read_csv("../Data/tcdml1920-rec-click-pred--training.csv"
                     , na_values = ['\\N','Withheld for privacy','nA','nan']
                     , dtype={"query_document_id": object, "first_author_id": object, "organization_id": object
                             , "recommendation_algorithm_id_used": object}
                    )

    dt_cols = ['request_received'
                   , 'response_delivered', 'time_recs_displayed', 'local_time_of_request', 'time_recs_recieved','time_recs_viewed'
              ]

    df = df[df['organization_id'] == str(org)]
    
    for i in dt_cols:
        df[i] = df[i].astype('datetime64[D]')
        
    ###### kaggle data
    dfkaggle = pd.read_csv("../Data/tcdml1920-rec-click-pred--test.csv"
                     , na_values = ['\\N','Withheld for privacy','nA','nan']
                     , dtype={"query_document_id": object, "first_author_id": object, "organization_id": object
                             , "recommendation_algorithm_id_used": object}
                    )

    for i in dt_cols:
        dfkaggle[i] = dfkaggle[i].astype('datetime64[D]')
        
    print("Completed read_data - ",datetime.datetime.now())
    
    return df, dt_cols, dfkaggle


# In[76]:


def cleaner1(df, dfkaggle, dt_cols):
    
    df2 = df.copy()
    dfkaggle2 = dfkaggle.copy()

    for i in dt_cols:
        df2[i+'_dow'] = df2[i].dt.day_name()
        dfkaggle2[i+'_dow'] = dfkaggle2[i].dt.day_name()
        df2[i+'_month'] = df2[i].dt.month_name()
        dfkaggle2[i+'_month'] = dfkaggle2[i].dt.month_name()
        df2[i+'_year'] = df2[i].dt.year
        dfkaggle2[i+'_year'] = dfkaggle2[i].dt.year
        df2[i+'_day'] = df2[i].dt.day
        dfkaggle2[i+'_day'] = dfkaggle2[i].dt.day
        df2 = df2.drop(columns = [i])
        dfkaggle2 = dfkaggle2.drop(columns = [i])

#     binary_cols = ['search_title','search_keywords','search_abstract']
#     binary_dict = {"yes":'1',"no":'0'}
#     for i in binary_cols:
#         df2[i] = np.where(df2[i].isna(),-1,np.where(df2[i] == 'yes',1,0))
#         df2[i] = df2[i].astype(int)
#     for i in binary_cols:
#         dfkaggle2[i] = np.where(dfkaggle2[i].isna(),-1,np.where(dfkaggle2[i] == 'yes',1,0))
#         dfkaggle2[i] = dfkaggle2[i].astype(int)
    
    drop_cols = ['clicks','ctr','recommendation_set_id','search_title','search_keywords','search_abstract','algorithm_class', 'cbf_parser']
    df2 = df2.drop(columns = drop_cols)
    dfkaggle2 = dfkaggle2.drop(columns = drop_cols)
    
    
##### Final Column List
    l1 = list(df2.dropna(axis = 1, how = 'all').columns)
    l2 = list(dfkaggle2.dropna(axis = 1, how = 'all').columns)
    l = [i for i in l1 if i in l2]
    l.append('set_clicked')
    
    print("Cleaner 1 - ",datetime.datetime.now())
    return df2[l], dfkaggle2[l]


# In[82]:


def split_val(df2):
    df3 = df2.copy()
    df3 = df3.reset_index().drop(columns = 'index')
    s = np.random.uniform(low = 0, high = df3.shape[0]-6, size = round(df3.shape[0]*0.20))
    s = s.round()
    dfv = df3.iloc[s,:]
    dft = df3.drop(s)
    print("Completed split val - ",datetime.datetime.now())
    return dfv, dft


# In[81]:


def undersampler(df2):
    us_val = random.randint(10,20)/1000 ########### IMPORTANT Parameter
    df2_0 = df2[df2['set_clicked'] == 0].reset_index()
    df2_1 = df2[df2['set_clicked'] == 1].reset_index()
    rows = round(df2_1.shape[0]/us_val)
    s = np.random.uniform(low = 0, high = df2_0.shape[0], size = rows)
    df2_0 = df2_0.iloc[s,:]
    df2 = df2_0.append(df2_1).drop(columns = 'index')
    print("Completed Undersampling - ",datetime.datetime.now())
    return df2, us_val


# In[83]:


def model(df4, df_val, dfk):
    
    X_train, X_test, Y_train, Y_test = train_test_split(df4.drop(columns = ['set_clicked'])
                                                        , df4['set_clicked'], test_size = 0.30 ##### change to lightgbm
#                                                         , random_state = 42
                                                       )
    
    print('Ones in train :',Y_train.sum(),'Ones in test:',Y_test.sum())

    rand = random.randint(1,2)
    
#################################### LGBM
    
#     if rand == 1:
        
#         params = {'boosting_type': 'gbdt',
#                   'max_depth' : 4,
#                   'objective': 'binary',
#                   'nthread': 4,
#                   'num_leaves': 64,
#                   'learning_rate': 0.001,
#                   'max_bin': 512,
#                   'subsample_for_bin': 200,
#                   'subsample': 1,
#                   'subsample_freq': 1,
#                   'colsample_bytree': 0.8,
#                   'reg_alpha': 1.2,
#                   'reg_lambda': 1.2,
#                   'min_split_gain': 0.5,
#                   'min_child_weight': 1,
#                   'min_child_samples': 5,
#                   'scale_pos_weight': 1,
#                   'num_class' : 1,
#                   'verbose': -1
#     #               'metric' : 'auc'
#                   }

# # #     making lgbm datasets for train and valid
#         d_train = lgbm.Dataset(X_train, Y_train)
#         d_valid = lgbm.Dataset(X_test, Y_test)

#         def lgb_f1_score(y_hat, data):
#             y_true = data.get_label()
#             y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
#             return 'f1', f1_score(y_true, y_hat), True

#         evals_result = {}

# #     training with early stop
# #   bst = lgbm.train(params, d_train, 5000, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=100)

# #     cat_vars_index = []
# #     for i in cat_vars:
# #         if i in X_train:
# #             cat_vars_index.append(X_train.columns.get_loc(i))


#         bst = lgbm.train(params, d_train, valid_sets=[d_valid, d_train], valid_names=['val', 'train'], feval=lgb_f1_score, evals_result=evals_result)

#################################### LGBM  
    
#################################### XGBoost
    
    if rand == 1:
    
        bst = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=4,
           min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
           objective='binary:logistic', reg_alpha=0, reg_lambda=1,
           scale_pos_weight=1, seed=0, silent=True, subsample=1) #, tree_method = 'hist'

        bst.fit(X_train,Y_train)

#     kfold = KFold(n_splits=10, random_state=42)  ##### Important Parameter
#     results = cross_val_score(bst, df_val.drop(columns = ['set_clicked']), df_val['set_clicked'], cv=kfold)

#################################### XGBoost


#################################### CatBoost
    
    if rand == 2:
        
        bst = CatBoostClassifier(eval_metric='F1',use_best_model=True, metric_period = 300, depth = 4)
        bst.fit(X_train,Y_train,eval_set=(X_test,Y_test)) ## cat_features = cat_vars_index

#################################### CatBoost

#     print("CV Score = ",results)
   
#     else:
        
#         bst = dt(max_depth = 4) # class_weight = {0:1,1:4}
#         bst.fit(X_train, Y_train)

    r = np.where(bst.predict(df_val.drop(columns = ['set_clicked'])) > 0.7, 1 ,0)
    
#     if rand in [2,4]:
#         kfold = KFold(n_splits=10, random_state=42)  ##### Important Parameter
#         results = cross_val_score(bst, df_val.drop(columns = ['set_clicked']), df_val['set_clicked'], cv=kfold)
#     else:
#         results = [0]

    results = []
    for i in range(10):
        df_val2 = shuffle(df_val)
        df_val3 = df_val2[0:int(df_val2.shape[0]*0.7)]
        rkf = bst.predict(df_val3.drop(columns = ['set_clicked']))
        results.append(accuracy_score(df_val3['set_clicked'], rkf))
        
    #Print accuracy
    acc_lgbm = accuracy_score(df_val['set_clicked'], r)

    print('Overall accuracy of model:', acc_lgbm, "   overall with only zeroes ", accuracy_score(df_val['set_clicked'], np.zeros(len(r))))
    
    check_increase = accuracy_score(df_val['set_clicked'], r) > accuracy_score(df_val['set_clicked'], np.zeros(len(r)))
#     print('Accuracy increased:',check_increase)
    #Print Area Under Curve
#     plt.figure()
    false_positive_rate, recall, thresholds = roc_curve(df_val['set_clicked'], r)
    roc_auc = auc(false_positive_rate, recall)
#     plt.title('Receiver Operating Characteristic (ROC)')
#     plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0,1], [0,1], 'r--')
#     plt.xlim([0.0,1.0])
#     plt.ylim([0.0,1.0])
#     plt.ylabel('Recall')
#     plt.xlabel('Fall-out (1-Specificity)')
#     plt.show()

    print('AUC score:', roc_auc)

    #Print Confusion Matrix
    plt.figure()
    cm = confusion_matrix(df_val['set_clicked'], r)
    # labels = ['No Default', 'Default']
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

    # lgbm.plot_metric(evals_result, metric='f1')
    
    f1 = f1_score(df_val['set_clicked'], r)
#     print(np.unique(bst.predict(dfk.drop(columns = ['set_clicked']))))
    rk = np.where(bst.predict(dfk.drop(columns = ['set_clicked'])) > 0.7, 1, 0)
    dfk['set_clicked'] = rk
    
    print("Completed Modelling - ",datetime.datetime.now())
    
    return acc_lgbm, f1, check_increase, X_train.columns, bst, dfk, results, accuracy_score(df_val['set_clicked'], np.zeros(len(r))), r, rand


# In[84]:


org = 1

# df, dt_cols, df_kaggle = read_data(org = org)

# df, d = process(df_w,run='train',dict_key={}) #train
# dfkaggle2 , d=process(df_kag,run='test',dict_key=d) #kaggle

r = []
max_f1 = 0
max_cv = 0
ones_list = []
l_r_val = []
    
for i in range(200):
    
    seed = random.randint(0,1000)
    np.random.seed(seed)
    print(seed)
    
    df2, dfkaggle2 = cleaner1(df, df_kaggle, dt_cols)

    dfv2, dft2 = split_val(df2)
    dft2, us_val = undersampler(dft2)
    
    encs_final_dicts = [{'user_id': 'CatBoostEncoder', 'session_id': 'CatBoostEncoder', 'query_identifier': 'CatBoostEncoder', 'query_detected_language': 'TargetEncoder', 'query_document_id': 'ColumnDropper', 'abstract_detected_language': 'OneHotEncoder', 'organization_id': 'WOEEncoder', 'application_type': 'CatBoostEncoder', 'item_type': 'ColumnDropper', 'app_lang': 'OneHotEncoder', 'country_by_ip': 'CatBoostEncoder', 'timezone_by_ip': 'TargetEncoder', 'recommendation_algorithm_id_used': 'OrdinalEncoder', 'request_received_dow': 'ColumnDropper', 'request_received_month': 'TargetEncoder', 'local_time_of_request_dow': 'ColumnDropper', 'local_time_of_request_month': 'OrdinalEncoder'}]


    for i in [encs_final_dicts[0]]: ### unlist and take all elements for full run
        dft3, dfv3, dfkaggle3, encoders_used = encode_all(df = dft2, dfv = dfv2, dfk = dfkaggle2, encoder_to_use = i)
        dft3.dropna(axis=1, how='all', inplace = True)
        dfv3 = dfv3[dft3.columns]
        dfkaggle3 = dfkaggle3[dft3.columns]
        
        imps_final_dict = [{'query_word_count': {'mvi': 'iterativeimputer', 'Indicator': False}, 'query_char_count': {'mvi': 'mean', 'Indicator': False}, 'abstract_word_count': {'mvi': 'mean', 'Indicator': False}, 'abstract_char_count': {'mvi': 'mean', 'Indicator': False}, 'hour_request_received': {'mvi': 'iterativeimputer', 'Indicator': False}, 'local_hour_of_request': {'mvi': 'most_frequent', 'Indicator': False}, 'recommendation_algorithm_id_used': {'mvi': 'deleterows', 'Indicator': False}, 'request_received_year': {'mvi': 'iterativeimputer', 'Indicator': False}, 'request_received_day': {'mvi': 'deleterows', 'Indicator': False}, 'local_time_of_request_month': {'mvi': 'deleterows', 'Indicator': False}, 'local_time_of_request_year': {'mvi': 'deleterows', 'Indicator': False}, 'local_time_of_request_day': {'mvi': 'most_frequent', 'Indicator': False}}]
        
        for j in [imps_final_dict[0]]: ### unlist and take all elements for full run
            dft4, dfv4, dfkaggle4, result = imputer(df = dft3, dfv = dfv3, dfk = dfkaggle3, imputer_dict = j)
            
#             dft4 = dft4.drop(columns = ['user_id', 'session_id', 'query_identifier','query_document_id'])
#             dfv4 = dfv4[dft4.columns]
#             dfkaggle4 = dfkaggle4[dft4.columns]

            accuracy, f1, check_increase, x_cols, bst, dfk, cv, zero_acc, r_val, rand = model(dft4, dfv4, dfkaggle4)
    #         if check_increase == True:
            ones_list.append(dfk['set_clicked'])
            if statistics.mean(cv) > max_cv:
                max_cv = statistics.mean(cv)
                dfk.to_csv("kaggle submission data.csv")
                
            l_r_val.append(r_val)
            rl = [org,i,j,accuracy,f1, check_increase, x_cols, rand, bst, 'importance', statistics.mean(cv), zero_acc, us_val, datetime.datetime.now()] ## bst.get_booster().get_score(importance_type="cover")
            r.append(rl)
            pd.DataFrame(r).to_csv("model results.csv", index = False)
            
pd.DataFrame(ones_list).to_csv("test multi ones jabref.csv")

