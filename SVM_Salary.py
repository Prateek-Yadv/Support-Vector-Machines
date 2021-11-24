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


string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native","Salary"]


# In[3]:


salary_train=pd.read_csv('C:/Users/prate/Downloads/Assignment/Support Vector Machines/SalaryData_Train.csv')
salary_train


# In[4]:


salary_test=pd.read_csv('C:/Users/prate/Downloads/Assignment/Support Vector Machines/SalaryData_Test.csv')
salary_test.head()


# In[5]:


salary_train.describe


# In[6]:


salary_test.describe


# In[7]:


salary_test.shape


# In[8]:


salary_train.shape


# In[9]:


salary_train.isnull().sum()


# In[10]:


salary_test.isnull().sum()


# In[11]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for i in string_columns:
    salary_train[i]=lb.fit_transform(salary_train[i])
    salary_test[i]=lb.fit_transform(salary_test[i])
        


# In[12]:


salary_test


# In[13]:


x_train = salary_train.iloc[:,0:13]
y_train = salary_train.iloc[:,13]
x_test = salary_test.iloc[:,0:13]
y_test = salary_test.iloc[:,13]


# In[14]:


##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
x_train = norm_func(x_train)
x_test =  norm_func(x_test)


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size = 0.3)


# In[16]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# ### Grid Search CV

# In[17]:


model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test) # Accuracy = 81%


# In[18]:


model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) # Accuracy = 84%


# In[19]:


model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) # Accuracy = 84%


# In[20]:


model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test) #Accuracy = 84%


# In[ ]:




