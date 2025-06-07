#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Titanic_train.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


# Data Cleaning
# checking Null values
df.isnull().sum()


# In[8]:


# Replacing null values of Age column by average
df['Age']=df['Age'].fillna(round(df['Age'].mean(),2))


# In[9]:


# irrelevant columns
df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)


# In[10]:


plt.title('Embarked count plot')
df['Embarked'].value_counts().plot(kind='bar')


# In[11]:


df['Embarked']=df['Embarked'].fillna('S')


# In[12]:


import myStats
for col_name in df.select_dtypes('number'):
    print(col_name)
    myStats.cal_stats(df[col_name])
    print('-'*50)


# In[13]:


for col_name in df.select_dtypes('number'):
    print(col_name)
    sns.displot(df[col_name])
    plt.show()


# In[14]:


for col_name in df.select_dtypes('object'):
    print(col_name)
    sns.countplot(df[col_name])
    plt.show()


# In[15]:


for col_name in df.select_dtypes('number'):
    print(col_name)
    sns.boxplot(df[col_name])
    plt.show()


# In[16]:


# correlation
sns.heatmap(df.corr(numeric_only=True), annot=True, vmin=-1, vmax=1, cmap='coolwarm')


# In[17]:


# one hot encoding
df=pd.get_dummies(df, dtype=int, drop_first=True)


# In[18]:


df.head()


# In[19]:


# adding new column as total Family
df['Total fam']=df['SibSp']+df['Parch']


# In[20]:


#X=input data, y=output
X=df.drop('Survived', axis=1)
y=df['Survived']


# In[21]:


#standardization
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)


# In[22]:


X=pd.DataFrame(X,columns=df.drop('Survived',axis=1).columns)


# In[23]:


X.head()


# In[24]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X,y)


# In[25]:


lr.coef_


# In[26]:


lr.intercept_


# In[27]:


# Loading test file
df_test=pd.read_csv('Titanic_test.csv')


# In[28]:


df_test.head()


# In[29]:


# irrelevant columns
df_test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)


# In[30]:


df_test['Total fam']=df['SibSp']+df['Parch']


# In[31]:


# one hot encoding
df_test=pd.get_dummies(df, dtype=int, drop_first=True)


# In[32]:


X_test=df_test.drop('Survived',axis=1)


# In[33]:


y_test=df_test['Survived']


# In[34]:


X_test.head()


# In[35]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_test=ss.fit_transform(X_test)


# In[36]:


X_test=pd.DataFrame(X_test,columns=df.drop('Survived',axis=1).columns)
X_test.head()


# In[37]:


y_test.head()


# In[38]:


y_pred=lr.predict(X_test)


# In[39]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




