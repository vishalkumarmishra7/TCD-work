import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pyod.models.iforest import IForest
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc, mean_absolute_error
import datetime
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import random
import category_encoders as ce
import math
import statistics
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("tcd-ml-1920-group-income-train.csv"
                 , na_values = ['nA','#N/A','#NUM!']
                )

drop_cols = ['Instance', 'Hair Color', 'Body Height [cm]' ,'Wears Glasses'] # , 'Crime Level in the City of Employement']
df['Yearly Income in addition to Salary (e.g. Rental Income)'] = df['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(" EUR","").astype(float)
df = df.drop(columns = drop_cols)

dfkaggle = pd.read_csv("tcd-ml-1920-group-income-test.csv"
                 , na_values = ['nA','#N/A','#NUM!']
                )

dfkaggle['Yearly Income in addition to Salary (e.g. Rental Income)'] = dfkaggle['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(" EUR","").astype(float)
dfkaggle = dfkaggle.drop(columns = drop_cols)

print("Completed read_data - ",datetime.datetime.now())




df2 = df.copy()
dfkaggle2 = dfkaggle.copy()

# clean_cols = ['Gender','Country','Housing Situation','University Degree'] #, 'Crime Level in the City of Employement']
# 
# for i in clean_cols:
#     df2[i] = np.where(df2[i] == '0', np.nan, df2[i])
#     dfkaggle2[i] = np.where(dfkaggle2[i] == '0', np.nan, dfkaggle2[i])
    
df2.columns = [i.replace("]","").replace("[","").replace(" ","_").replace(")","").replace("(","") for i in df2.columns]
dfkaggle2.columns = df2.columns

df2['Profession2'] = df2['Profession'].str.slice(0,5)
dfkaggle2['Profession2'] = dfkaggle2['Profession'].str.slice(0,5)

# df2['Total_Yearly_Income_EUR'] = np.log(df2['Total_Yearly_Income_EUR'])

for i in df2:
    if df2[i].dtype.kind not in 'bifuc':
        df2[i] = df2[i].str.lower()
        df2[i] = df2[i].str.strip()
        dfkaggle2[i] = dfkaggle2[i].str.lower()
        dfkaggle2[i] = dfkaggle2[i].str.strip()
        
l = ['Housing_Situation', 'Age'] # 'Housing_Situation', 'Satisfation_with_employer', 'Gender', 'Age', 'Country', 'University_Degree'
# 
# for i in l:
#     for j in l:
#         if i != j:
#             df2[i+j] = df2[i].astype(str) + df2[j].astype(str)
#             dfkaggle2[i+j] = dfkaggle2[i].astype(str) + dfkaggle2[j].astype(str)

df2['is_senior'] = np.where(df2['Profession'].str.contains("senior"),1,0)
dfkaggle2['is_senior'] = np.where(dfkaggle2['Profession'].str.contains("senior"),1,0)

df2['is_manager'] = np.where(df2['Profession'].str.contains("manager"),1,0)
dfkaggle2['is_manager'] = np.where(dfkaggle2['Profession'].str.contains("manager"),1,0)

print("Cleaner 1 - ",datetime.datetime.now())



df3 = df2.copy()
df3 = df3.reset_index(drop=True)
s = np.random.uniform(low = 0, high = df3.shape[0]-1, size = round(df3.shape[0]*0.05))
s = s.round()
dfv = df3.iloc[s,:]
dft = df3.drop(s)
print("Completed split val - ",datetime.datetime.now())


l = []
for i in dft:
    if dft[i].dtype.kind not in 'bifuc':
        f_enc = dft[i].value_counts(dropna=False, normalize=True).to_dict()
        m_enc = dft.groupby(i)['Total_Yearly_Income_EUR'].mean()
        v_enc = dft.groupby(i)['Total_Yearly_Income_EUR'].std()
        
        dft[i+'_f'] = dft[i].map(f_enc)
        dfv[i+'_f'] = dfv[i].map(f_enc)
        dfkaggle2[i+'_f'] = dfkaggle2[i].map(f_enc)
        
        dft[i+'_m'] = dft[i].map(m_enc)
        dfv[i+'_m'] = dfv[i].map(m_enc)
        dfkaggle2[i+'_m'] = dfkaggle2[i].map(m_enc)
        
        dft[i+'_v'] = dft[i].map(v_enc)
        dfv[i+'_v'] = dfv[i].map(v_enc)
        dfkaggle2[i+'_v'] = dfkaggle2[i].map(v_enc)
        
#         dft = dft.drop(columns = i)
#         dfv = dfv.drop(columns = i)
#         dfkaggle2 = dfkaggle2.drop(columns = i)
        
        l.append(i)
        
encoder=ce.CatBoostEncoder(cols=l,return_df=1,drop_invariant=1,handle_missing=False,sigma=None,a=2)
encoder.fit(X=dft,y=dft['Total_Yearly_Income_EUR'])
dft=encoder.transform(dft)
dfv=encoder.transform(dfv)
dfkaggle2=encoder.transform(dfkaggle2)
            
        
imp = IterativeImputer(
                    max_iter = 3
                   , estimator = ExtraTreesRegressor()
                   , n_nearest_features = 5
                  )

dfc = dft.copy()
dft = dft.drop(columns = 'Total_Yearly_Income_EUR')

dfvc = dfv.copy()
dfv = dfv.drop(columns = 'Total_Yearly_Income_EUR')

dfkc = dfkaggle2.copy()
dfkaggle2 = dfkaggle2.drop(columns = 'Total_Yearly_Income_EUR')

dfcolumns = dft.columns
imp.fit(dft)

dft = pd.DataFrame(imp.transform(dft))
dft.columns = dfcolumns
dfv = pd.DataFrame(imp.transform(dfv))
dfv.columns = dfcolumns
dfkaggle2 = pd.DataFrame(imp.transform(dfkaggle2))
dfkaggle2.columns = dfcolumns

dft['Total_Yearly_Income_EUR'] = np.array(dfc['Total_Yearly_Income_EUR'])
dfv['Total_Yearly_Income_EUR'] = np.array(dfvc['Total_Yearly_Income_EUR'])
dfkaggle2['Total_Yearly_Income_EUR'] = np.nan


print('Completed Imputing')


#### feature creation

for i in dft.drop(columns = 'Total_Yearly_Income_EUR'):
    dft[i+'_log'] = np.log(np.where(dft[i] == 0, 0.0005, dft[i]))
    dfv[i+'_log'] = np.log(np.where(dfv[i] == 0, 0.0005, dfv[i]))
    dfkaggle2[i+'_log'] = np.log(np.where(dfkaggle2[i] == 0, 0.0005, dfkaggle2[i]))
    

# dft = dft.fillna(0)
# dfv = dfv.fillna(0)
# dfkaggle2 = dfkaggle2.fillna(0)

cols = dft.drop(columns = 'Total_Yearly_Income_EUR').columns
scaler = StandardScaler()
scaler.fit(dft.drop(columns = 'Total_Yearly_Income_EUR'))

dftc = dft.copy()
dft = pd.DataFrame(scaler.transform(dft.drop(columns = 'Total_Yearly_Income_EUR')))
dft.columns = cols
dft['Total_Yearly_Income_EUR'] = np.array(dftc['Total_Yearly_Income_EUR'])

dfvc = dfv.copy()
dfv = pd.DataFrame(scaler.transform(dfv.drop(columns = 'Total_Yearly_Income_EUR')))
dfv.columns = cols
dfv['Total_Yearly_Income_EUR'] = np.array(dfvc['Total_Yearly_Income_EUR'])

dfkc = dfkaggle2.copy()
dfkaggle2 = pd.DataFrame(scaler.transform(dfkaggle2.drop(columns = 'Total_Yearly_Income_EUR')))
dfkaggle2.columns = cols
dfkaggle2['Total_Yearly_Income_EUR'] = np.array(dfkc['Total_Yearly_Income_EUR'])


print("Completed Scaling - ",datetime.datetime.now())


bst = LGBMRegressor(boosting_type='gbdt'
                    , max_depth=9
                    , learning_rate=0.05
                    , n_estimators=10000
                    , objective='regression'
                    , min_split_gain=0.0
                    , min_child_weight=0.001
                    , min_child_samples=20
                    , num_leaves = 150
                    , n_jobs=-1
                    , silent=True
                    , importance_type='split'
                    , device = 'gpu'
                    , gpu_platform_id = 0
                    , gpu_device_id = 0)

dft2 = dft.sample(frac = 0.5)

bst.fit(dft2.drop(columns = ['Total_Yearly_Income_EUR']), dft2['Total_Yearly_Income_EUR']
        , eval_set=(dft2.drop(columns = ['Total_Yearly_Income_EUR']), dft2['Total_Yearly_Income_EUR'])
        , verbose = 500)

                       
dft['Total_Yearly_Income_EUR_1'] = bst.predict(dft.drop(columns = ['Total_Yearly_Income_EUR']))
dfv['Total_Yearly_Income_EUR_1'] = bst.predict(dfv.drop(columns = ['Total_Yearly_Income_EUR']))
dfkaggle2['Total_Yearly_Income_EUR_1'] = bst.predict(dfkaggle2.drop(columns = ['Total_Yearly_Income_EUR']))

    
bst = xgb.XGBRegressor(
                        base_score=0.5
                        , colsample_bylevel=1
                        , colsample_bytree=1
                        , gamma=0
                        , learning_rate=0.1
                        , max_delta_step=0
                        , max_depth = 10
                        , min_child_weight=1
                        , missing = None
                        , n_estimators = 5000
                        , nthread=-1
                        , objective = 'reg:squarederror'
                        , eval_metric = 'mae'
                        , reg_alpha = 0
                        , reg_lambda = 1
                        , scale_pos_weight = 1
                        # , seed = 0
                        , silent = False
                        , subsample=1
                        , verbose = 1
                        , tree_method = 'gpu_hist'
                        , gpu_id = 0)

dft2 = dft.sample(frac = 0.5)

bst.fit(dft2.drop(columns = ['Total_Yearly_Income_EUR']), dft2['Total_Yearly_Income_EUR'])

dft['Total_Yearly_Income_EUR_2'] = bst.predict(dft.drop(columns = ['Total_Yearly_Income_EUR']))
dfv['Total_Yearly_Income_EUR_2'] = bst.predict(dfv.drop(columns = ['Total_Yearly_Income_EUR']))
dfkaggle2['Total_Yearly_Income_EUR_2'] = bst.predict(dfkaggle2.drop(columns = ['Total_Yearly_Income_EUR']))

bst = CatBoostRegressor(eval_metric='MAE'
                        # , use_best_model=True
                        , metric_period = 5000
                        , depth = 9
                        , task_type = "GPU"
                        , devices = '0:1'
                        , num_boost_round = 15000)
       
bst.fit(dft.drop(columns = ['Total_Yearly_Income_EUR']), dft['Total_Yearly_Income_EUR']
        , eval_set=(dft.drop(columns = ['Total_Yearly_Income_EUR']), dft['Total_Yearly_Income_EUR']))
        
# r = np.exp(bst.predict(dfv.drop(columns = ['Total_Yearly_Income_EUR'])))
# acc_score = mean_absolute_error(np.exp(dfv['Total_Yearly_Income_EUR']), r)

r = bst.predict(dfv.drop(columns = ['Total_Yearly_Income_EUR']))
acc_score = mean_absolute_error(dfv['Total_Yearly_Income_EUR'], r)
print("Accuracy Validation - ",acc_score)

r = bst.predict(dft.drop(columns = ['Total_Yearly_Income_EUR']))
acc_score = mean_absolute_error(dft['Total_Yearly_Income_EUR'], r)
print("Accuracy Training- ",acc_score)


# rk = np.exp(bst.predict(dfkaggle2.drop(columns = ['Total_Yearly_Income_EUR'])))

rk = bst.predict(dfkaggle2.drop(columns = ['Total_Yearly_Income_EUR']))
dfkaggle2['Total_Yearly_Income_EUR'] = rk

dfkaggle2[['Total_Yearly_Income_EUR']].to_csv("Kaggle submission Stacking.csv")

print("Completed Modelling - ",datetime.datetime.now())
