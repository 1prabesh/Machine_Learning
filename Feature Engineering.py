#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


df=pd.read_csv('Ads.csv',usecols=['Age','EstimatedSalary','Purchased'])
df.head()


# In[25]:


X_train,X_test, y_train,y_test=train_test_split(df.drop('Purchased', axis=1),df['Purchased'], test_size=0.3, random_state=0)
X_train.shape


# In[16]:


X_test.shape


# In[18]:


# creating object of standard scaler for scaling the data


# In[26]:


scaler=StandardScaler()


# In[20]:


# now fit the train data to scaler object to learn the parameters from it


# In[27]:


scaler.fit(X_train) # it is calculating mean and standard deviation of the columns (i.e, age and estimated salary)


# In[28]:


scaler.mean_


# In[29]:


# now use the same scaler which has already calculated mean and standard deviation to normalize the column (X-X bar/ sigma)


# In[30]:


X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[34]:


X_train_scaled


# In[35]:


# since our output is numpy array we want to convert it to dataframe from pandas


# In[38]:


X_train_scaled=pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_scaled


# In[44]:


np.round(X_train.describe(),1)


# In[45]:


np.round(X_train_scaled.describe(),1) # this has converted mean to 0 and std to 1


# In[46]:


# verifying the effect of scaling in the data


# In[49]:


sns.scatterplot(x=X_train['Age'],y=X_train['EstimatedSalary'])


# In[52]:


sns.scatterplot(x=X_train_scaled['Age'],y=X_train_scaled['EstimatedSalary'],color='red')


# ## Categorical encoding

# In[7]:


dataframe=pd.read_csv('customer.csv',usecols=['review','education','purchased'])
dataframe.head()


# In[8]:


#in the first phase we will use label incoder and ordinal incoder 


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(dataframe.drop('purchased',axis=1),dataframe['purchased'],test_size=0.2)


# In[21]:


X_train.head()


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])
OE.fit(X_train)
X_train_fitted=OE.transform(X_train)
X_test_fitted=OE.transform(X_test)


# In[25]:


X_train_fitted.shape


# In[19]:


X_test_fitted.shape


# In[26]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
LE.fit(y_train)
y_train_fitted=LE.transform(y_train)
y_test_fitted=LE.transform(y_test)


# In[27]:


y_train_fitted


# In[29]:


y_train_fitted.shape


# In[32]:


type(y_train)


# In[33]:


type(y_train_fitted)


# In[36]:


onecol=pd.DataFrame(y_train,y_train_fitted)


# In[39]:


y_train_fitted


# In[40]:


type(y_train_fitted)


# ## One Hot Encoding

# In[41]:


data2=pd.read_csv('cars.csv')
data2.sample(5)


# In[62]:


data2['brand'].value_counts()


# In[77]:


### using pandas one hot encoding
data2['fuel'].value_counts()


# In[70]:


pd.get_dummies(data2, columns=['fuel','owner'])


# In[69]:


#now drop one column to remove dummy trap


# In[72]:


pd.get_dummies(data2, columns=['fuel','owner'], drop_first=True)


# ### following the reliable way of one hot encoding

# In[88]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data2.drop('selling_price', axis=1),data2['selling_price'],test_size=0.2,random_state=0)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(drop='first') # why using drop? to remove colinearlity as learned before this drop first colum from both 


# In[79]:


#fitting ohe to X_train data


# In[89]:


X_train_fitted=ohe.fit_transform(X_train[['fuel','owner']]).toarray()


# In[90]:


X_train_fitted


# In[91]:


X_test_fitted=ohe.transform(X_test[['fuel','owner']]).toarray()
X_test_fitted


# In[92]:


X_train[['brand','km_driven']].values


# In[93]:


np.hstack((X_train[['brand','km_driven']].values,X_train_fitted)).shape


# In[ ]:




