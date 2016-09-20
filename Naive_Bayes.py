
# coding: utf-8

# In[32]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic(u'matplotlib inline')
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score,recall_score,roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, StratifiedKFold


# In[33]:

import random
random.seed(2016)


# # Loading the dataset

# In[34]:

data=pd.read_csv('data1.csv')


# In[35]:

X=data.drop('ARR_DEL15',axis=1)

Y=data.ARR_DEL15


# # Train Test split
# Spliting the dataset into training and testing data using the train_test_split method

# In[36]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=168)


# # Logistic Regression
# Performing logistic regression on the training data and evaluating it on the test data.Performed the Classification using grid search method and cross-validation with 10 fold

# In[37]:

r_scorer = make_scorer(roc_auc_score)
gaussian = GaussianNB()
param_grid = {}
grid_ob = GridSearchCV(gaussian,param_grid,scoring=r_scorer,cv=10)
grid_ob=grid_ob.fit(X_train, y_train)
clf=grid_ob.best_estimator_
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print "Area under curve for Logistic predection scores-",roc_auc_score(y_test, predictions)


# In[38]:


gaussian.fit(X_train, y_train)
predictions = gaussian.predict(X_test)
print(roc_auc_score(y_test, predictions))


# # Confusion Matrix

# In[39]:

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, predictions)
print "Confusion Matrix ", confusion_matrix

print ("          Predicted")
print ("         |  0  |  1  |")
print ("         |-----|-----|")
print ("       0 | %3d | %3d |" % (cm[0, 0],
                                   cm[0, 1]))
print ("Actual   |-----|-----|")
print ("       1 | %3d | %3d |" % (cm[1, 0],
                                   cm[1, 1]))
print ("         |-----|-----|")


# # Classification Report showing the Precision and Recall values for this model

# In[31]:

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# # Ploting Area Under Curve for Logistic Regression Model

# In[30]:

from sklearn.metrics import roc_curve, roc_auc_score

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (recall)")

#decision_function = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, predictions)
acc = clf.score(X_test, y_test)
auc = roc_auc_score(y_test, predictions)
plt.plot(fpr, tpr, label="area under curve for NB:%.2f" % ( auc), linewidth=3)
plt.legend(loc="best")


# In[ ]:




# In[ ]:




# In[ ]:



