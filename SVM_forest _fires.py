#!/usr/bin/env python
# coding: utf-8

# In[1]:


# SVM Classification
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


ff=pd.read_csv('C:/Users/prate/Downloads/Assignment/Support Vector Machines/forestfires.csv')
ff.head()


# In[3]:


ff.describe


# In[4]:


ff.shape


# In[5]:


ff.isnull().sum()


# In[6]:


from sklearn.preprocessing import LabelEncoder
label_encoder =LabelEncoder()
ff['month']= label_encoder.fit_transform(ff['month']) 
ff['day']= label_encoder.fit_transform(ff['day'])
ff['size_category']= label_encoder.fit_transform(ff['size_category'])
ff


# In[7]:


x=ff.iloc[:,2:30]
y=ff.iloc[:,30]


# In[8]:


x.head()


# In[9]:


y.head()


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)


# In[11]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# ### Grid Search CV

# In[12]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(x_train,y_train)


# In[13]:


gsv.best_params_ , gsv.best_score_ 


# In[14]:


clf = SVC(C= 15, gamma = 50)
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[ ]:




