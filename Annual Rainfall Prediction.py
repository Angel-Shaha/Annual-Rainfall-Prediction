#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[32]:


data=pd.read_csv(r'C:\Users\HP\Downloads\AllIndiaRainfall.csv')


# In[33]:


X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
Y=data.iloc[:,14].values


# In[34]:


Y=Y.reshape(-1,1)


# In[35]:


#cleansing
imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')


# In[36]:


X=imp.fit_transform(X)
Y=imp.fit_transform(Y)


# In[37]:


#encoding
label=LabelEncoder()
X[:,0]=label.fit_transform(X[:,0])


# In[38]:


#splitting dataset
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[42]:


#training
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
figure,axis=plt.subplots(2)
axis[0].plot(Y_test,color='green')
axis[0].set_title('Real values of annual rainfall')
axis[1].plot(Y_pred,color='red')
axis[1].set_title('Predicted values of annual rainfall')
plt.show()


# In[43]:


#obtaining the dataframe
df=np.concatenate((Y_test,Y_pred),axis=1)
dataframe=pd.DataFrame(df,columns=['Real annual rainfall','Predicted annual rainfall'])
print(dataframe)


# In[44]:


#accuracy score
ac=reg.score(X_test,Y_test)*100
print('Accuracy=',ac)


# In[ ]:




