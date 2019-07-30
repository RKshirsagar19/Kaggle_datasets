#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()


# In[10]:


print(cancer['DESCR'])
print(cancer['feature_names'])


# In[13]:


print(cancer['data'].shape)


# In[14]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# In[15]:


df_cancer.head()


# In[16]:


df_cancer.tail()


# In[17]:


df_cancer.describe()


# In[20]:


sns.pairplot(df_cancer, hue='target', vars=['mean radius','mean texture','mean area', 'mean perimeter','mean smoothness'])


# In[21]:


sns.countplot(df_cancer['target'])


# In[22]:


sns.scatterplot(x='mean area',y='mean smoothness', hue='target', data=df_cancer)


# In[23]:


plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot=True)


# In[28]:


##Model training

X= df_cancer.drop(['target'], axis=1)


# In[26]:


y=df_cancer['target']


# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


X_train.shape


# In[33]:


y_train.shape


# In[34]:


X_test.shape


# In[37]:


from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()


# In[38]:


svc_model.fit(X_train, y_train)


# In[39]:


y_pred= svc_model.predict(X_test)


# In[40]:


y_pred


# In[42]:


cm=confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot=True)


# In[43]:


##Improving the model

min_train = X_train.min()

range_train = (X_train-min_train).max()

X_train_scaled = (X_train-min_train)/range_train


# In[45]:


sns.scatterplot(x=X_train['mean area'], y=X_train['mean smoothness'], hue=y_train)


# In[46]:


sns.scatterplot(x=X_train_scaled['mean area'], y=X_train_scaled['mean smoothness'], hue=y_train)


# In[47]:


min_test = X_test.min()

range_test = (X_test-min_test).max()

X_test_scaled = (X_test-min_test)/range_test


# In[54]:


svc_model.fit(X_train_scaled,y_train)

y_pred= svc_model.predict(X_test_scaled)


# In[55]:


cm=confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot=True)


# In[57]:


print(classification_report(y_test, y_pred))


# In[58]:


param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(SVC(), param_grid, refit=True, verbose=4)

grid.fit(X_train_scaled, y_train)


# In[59]:


grid.best_params_


# In[60]:


grid_pred= grid.predict(X_test_scaled)


cm= confusion_matrix(y_test, grid_pred)

sns.heatmap(cm, annot=True)


# In[61]:


print(classification_report(y_test, y_pred))


# In[ ]:




