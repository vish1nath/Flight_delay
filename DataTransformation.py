
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
pd.options.display.max_columns=100
import matplotlib.pyplot as plt


# In[2]:

Down=pd.read_csv('flight.csv', index_col=None)


# In[3]:

Down


# In[5]:

df=Down[Down['ARR_DELAY_NEW'].notnull()]


# In[6]:

df=df[['UNIQUE_CARRIER','ORIGIN_STATE_ABR','DEST_STATE_ABR','DEP_DELAY_NEW','DISTANCE','ARR_DEL15']]


# In[7]:

carrier=pd.get_dummies(df.UNIQUE_CARRIER,prefix='CARRIER')
source=pd.get_dummies(df.ORIGIN_STATE_ABR,prefix='SOURCE')
dest=pd.get_dummies(df.DEST_STATE_ABR,prefix='DESTINATION')


# In[8]:

df=pd.concat([df,carrier,source,dest],axis=1)


# In[9]:

df=df.drop(['UNIQUE_CARRIER','ORIGIN_STATE_ABR','DEST_STATE_ABR'],axis=1)


# In[10]:

df.to_csv('data1.csv')


# In[ ]:




# In[ ]:



