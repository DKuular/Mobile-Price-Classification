#!/usr/bin/env python
# coding: utf-8

# ЗАДАЧА

# Найти наилучшую модель для составления прогноза ценового класса мобильных телефонов.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from IPython.display import display
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import LinearSVC


# In[2]:


train=pd.read_csv('/Users/deniskuular/Desktop/Tests/Python Projects/Mobile Price/train.csv')
train


# In[3]:


print(train.info())
print(train.shape)


# In[4]:


train.describe()


# In[5]:


train.loc[train.duplicated()]


# In[6]:


X=train.iloc[:, :20]
Y=train.iloc[:, 20:]


# In[7]:


X


# In[8]:


train.price_range.value_counts()


# In[9]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(train.corr(),  cmap='Greens')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12); 


# In[10]:


X_scaled=StandardScaler().fit_transform(X)

models = []
models.append(('LR', LogisticRegression(multi_class='multinomial'))) #для мультиклассовой классифbкации
models.append(('KNN', KNeighborsClassifier(n_neighbors=10)))
models.append(('CART', DecisionTreeClassifier(max_depth =100, criterion='gini', splitter='best')))
models.append(('RF', RandomForestClassifier(max_depth =100, criterion='gini')))

names = []
results=[]
for name, model in models:
    result=cross_val_score(model, X_scaled, np.ravel(Y), scoring="accuracy").mean()
    names.append(name)
    results.append(result)
    print('Результат перекрестной проверки %s : %f ' % (name, result))
  


# In[11]:


sns.set_style(style="dark") 

x = names
y = results

sns.barplot(x, y)
plt.show()


# По результатам перекрестной проверки, высокий результат нам показала логистическая регрессия. Соответственно, на основе этой модели мы составим прогноз.

# In[12]:


test=pd.read_csv('/Users/deniskuular/Desktop/Tests/Python Projects/Mobile Price/test.csv', index_col='id')
print(test)
print("Размерность тестового набора: ", test.shape)


# In[13]:


X_scaled=StandardScaler().fit_transform(X)
lr=LogisticRegression(multi_class='multinomial').fit(X_scaled,np.ravel(Y))
test_scaled=StandardScaler().fit_transform(test)
y_pred=lr.predict(test_scaled)
y_pred


# In[14]:


test['price_range']=y_pred
test


# In[16]:


sns.catplot(x=test['price_range'], y=test['int_memory'], data=test)


# In[17]:


sns.catplot(x=test['price_range'], y=test['ram'], data=test)


# In[18]:


sns.catplot(x=test['price_range'], y=test['fc'], data=test)


# In[ ]:




