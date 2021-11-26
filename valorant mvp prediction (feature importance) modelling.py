#!/usr/bin/env python
# coding: utf-8

# Regression models are used since we have to predict Average combat score- since MOST VALUBALE PLAYER mainly depends on what ACS is of the player. Since it is a continuous score, only regression models will be used

# ## Linear regression 
# 

# Linear regression modelling on dataset of players from all regions with different variables recorded in their performance in different competitive matches so far.

# In[569]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Lasso
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
plt.style.use("ggplot")
import xgboost as xgb
from sklearn import metrics
import lightgbm as ltb


# In[570]:


#importing the dataset
dataset=pd.read_csv('pdata.csv')
dataset.shape


# In[571]:


#displaying the dataset
dataset.describe
dataset


# In[572]:


#since player name is ID, we drop it from the dataset for training and testing
x1=dataset.drop('Player name',axis=1)
x=x1.astype(float)
x=x1.drop('acs',axis=1)
x


# In[573]:


#as linear regression is based on dependent and independent variables- ACS is dependent variable
y=x1['acs']
y


# In[574]:


plt.plot(x,y)
plt.xlabel("features")
plt.ylabel("acs")


# In[575]:


print(y.shape)
print(x.shape)
x=np.nan_to_num(x)


# In[576]:


# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 20)


# In[577]:


# importing module
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(x_train,y_train)


# In[578]:


y_prediction =  LR.predict(x_test)
print(y_prediction)


# In[579]:


#importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_prediction)
print("r2 score is ",score)
print("mean squared error is",mean_squared_error(y_test,y_prediction))
print("root mean squared error is",np.sqrt(mean_squared_error(y_test,y_prediction)))


# In[580]:


lrslop=LR.intercept_
lrcoef=LR.coef_
print(lrcoef)
print(lrslop)


# In[581]:


y_prediction=LR.predict(x_test)


# In[582]:


df = pd.DataFrame({'Actual': y_test,'Predicted': y_prediction})
df


# In[583]:


df1 = df.head(50)
df1.plot(kind='bar',figsize=(32,20))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[584]:


df=dataset.columns
print(df)
col=df.tolist()
print(col)
col.remove('Player name')
col.remove('acs')


# In[585]:


plt.figure(figsize=(20,10))
plt.title('Linear Regression Feature Importance')
plt.xticks(rotation=80)
plt.bar(col,lrcoef,color='maroon')


# ## Lasso regression

# In[586]:


import random
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10


# In[587]:


plt.plot(x,y,'.')


# In[588]:


# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 20)


# In[589]:


from sklearn.linear_model import LassoCV
#setting up model on train set
lasso_model = Lasso().fit(x_train,y_train)


# In[590]:


lasso_model.intercept_


# In[591]:


lasso_model.coef_


# In[592]:


lasso_model.score(x_test, y_test), lasso_model.score(x_train, y_train)


# 

# In[593]:


plt.figure(figsize=(20,10))
plt.title('Lasso regression Feature Importance')

plt.bar(col,lasso_model.coef_)


# In[594]:


lasso = Lasso()
coefs = []
alphas = np.random.randint(0,2000,200)
for a in alphas:
    lasso.set_params(alpha = a)
    lasso.fit(x_train,y_train)
    coefs.append(lasso.coef_)


# In[595]:


coefvsal = plt.gca()

coefvsal.plot(alphas, coefs,'--')
coefvsal.set_xscale("log")


# In[596]:


#prediction for test and train sets
print(lasso_model.predict(x_test))
prin(lasso_model.predict(x_train))


# In[ ]:


#RMSE value of predictions
y_prediction = lasso_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_prediction))


# In[ ]:


#R2 score of the set
r2_score(y_test, y_prediction)


# In[ ]:


df = pd.DataFrame({'Actual': y_test,'Predicted':y_prediction})
df


# In[ ]:


df1 = df.head(50)
df1.plot(kind='bar',figsize=(32,20))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')
plt.show()


# ## Random forest regression
# 

# In[ ]:


print(y.shape)
print(x.shape)
x=np.nan_to_num(x)


# In[597]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[643]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)
print(x_train)
print(y_train)
print(x_test)
print(y_test)


# In[644]:


plt.plot(x_train,y_train,'.',color='yellow')
plt.xlabel("features")
plt.ylabel("acs")


# In[645]:


plt.plot(x_test,y_test,'.',color='red')
plt.xlabel("features")
plt.ylabel("acs")


# In[646]:


rf = RandomForestRegressor(n_estimators=500,random_state=42, max_depth=5)
model=rf.fit(x_train, y_train)
print(model)


# In[647]:


predictions = rf.predict(x_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))


# In[648]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[649]:


importance=rf.feature_importances_
importance


# In[650]:


df=dataset.columns
print(df)
col=df.tolist()
print(col)
col.remove('acs')
col.remove('Player name')


# In[651]:


plt.bar(col, importance)
plt.title("Random Forest Feature importance")


# In[652]:


df = pd.DataFrame({'Actual': y_test,'Predicted': predictions})
df


# In[653]:


df1 = df.head(50)
df1.plot(kind='bar',figsize=(32,20))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ## XGBOOST regression
# 

# In[654]:


dataset=pd.read_csv('pdata.csv')
x1=dataset.drop('Player name',axis=1)
x1=x1.astype(float)
x=x1.drop('acs',axis=1)
x
y=x1['acs']
y


# In[610]:


data_dmatrix = xgb.DMatrix(data=x,label=y)


# In[655]:


# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[656]:


plt.plot(x_train,y_train,'.',color='black')
plt.xlabel("features")
plt.ylabel("acs")


# In[657]:


plt.plot(x_test,y_test,'.',color='green')
plt.xlabel("features")
plt.ylabel("acs")


# In[658]:


model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 500)
model.fit(x_train, y_train)
print(); print(model)


# In[659]:


y_train=np.nan_to_num(y_train)
x_train=np.nan_to_num(x_train)
y_test=np.nan_to_num(y_test)
x_test=np.nan_to_num(x_test)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
x_train=x_train.astype(int)
y_test=y_test.astype(int)


# In[660]:


predicted_y = model.predict(x_test)
print (predicted_y)


# In[661]:


df = pd.DataFrame({'Actual': y_test,'Predicted': predicted_y})
df


# In[662]:


df1 = df.head(50)
df1.plot(kind='bar',figsize=(32,20))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[663]:


rmse = np.sqrt(mean_squared_error(y_test, predicted_y))
print("RMSE: %f" % (rmse))


# In[664]:


params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=1000,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[665]:


cv_results.head()


# In[666]:


print((cv_results["test-rmse-mean"]).tail(1))


# In[667]:


xg_reg = xgb.train(params, dtrain=data_dmatrix, num_boost_round=10)


# In[668]:


xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()


# ## accuracy of this model
# 

# In[669]:


#Accuracy
from sklearn.model_selection import cross_val_score, KFold

scores = cross_val_score(model, x_train, y_train,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(model, x_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# ## lightGBM regression model

# In[670]:


data = pd.read_csv('pdata.csv')
print(data.head())
# define input and output feature
x1=data.drop('Player name',axis=1)
x1=x1.astype(float)
x=x1.drop('acs',axis=1)
x
y=x1['acs']
y


# In[671]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()     
le.fit_transform(y) 


# In[679]:


# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[680]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[681]:


y_train=np.nan_to_num(y_train)
x_train=np.nan_to_num(x_train)
y_test=np.nan_to_num(y_test)
x_test=np.nan_to_num(x_test)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
x_train=x_train.astype(int)
y_test=y_test.astype(int)


# In[682]:


import lightgbm as lgb
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


# In[683]:


model = ltb.LGBMRegressor(colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 500)
model.fit(x_train, y_train)
print(); print(model)

expected_y  = y_test
predicted_y = model.predict(x_test)


# In[684]:


print(metrics.r2_score(expected_y, predicted_y))
print(metrics.mean_squared_log_error(expected_y, predicted_y))


# In[685]:


plt.figure(figsize=(10,10))
sns.regplot(expected_y, predicted_y, fit_reg=True, scatter_kws={"color": "black" },line_kws={"color": "red"})


# In[686]:


df=data.columns.drop('Player name','acs')
print(df)
col=df.tolist()
col.remove('acs')
print(col)


# ## Feature importance in LightGBM

# In[687]:


ltb.plot_importance(model)
plt.title("LightGBM feature importance")
plt.xlabel('score')
plt.ylabel('features')


# In[688]:


df = pd.DataFrame({'Actual': y_test,'Predicted': predicted_y})
df


# In[689]:


df1 = df.head(50)
df1.plot(kind='bar',figsize=(32,20))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[690]:


lgb.plot_tree(model,figsize=(30,40))


# ## accuracy of this model
# 

# In[691]:


print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Final accuracy {:.4f}'.format(model.score(x_test,y_test)))


# In[ ]:





# In[ ]:




