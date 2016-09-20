
# coding: utf-8

# In[29]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic(u'matplotlib inline')
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import make_scorer, accuracy_score,recall_score,roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, StratifiedKFold


# In[30]:

import random
random.seed(2016)


# # Loading the dataset

# In[118]:

data=pd.read_csv('data1.csv')


# In[119]:

X=data.drop('ARR_DEL15',axis=1)
Y=data.ARR_DEL15


# # Train Test split
# Spliting the dataset into training and testing data using the train_test_split method

# In[120]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=168)


# # Support Vector Machine
# Performing Linear SVC on the training data and evaluating it on the test data.Performed the Classification using grid search method and cross-validation with 10 fold

# In[130]:

r_scorer = make_scorer(recall_score)
svc = LinearSVC(penalty='l2')
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10,100],
              }
grid_ob = GridSearchCV(svc, param_grid,scoring=r_scorer,cv=10)
grid_ob=grid_ob.fit(X_train, y_train)
svc=grid_ob.best_estimator_
svc.fit(X_train,y_train)
predictions = svc.predict(X_test)
print "Area under curve for SVM predection scores-",roc_auc_score(y_test, predictions)


# # Confusion Matrix

# In[131]:

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

# In[132]:

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# # Ploting Area Under Curve for SVM Model

# In[106]:

from sklearn.metrics import roc_curve, roc_auc_score

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (recall)")

decision_function = svc.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, predictions)
acc = svc.score(X_test, y_test)
auc = roc_auc_score(y_test, predictions)
plt.plot(fpr, tpr, label="acc:%.2f auc:%.2f" % (acc, auc), linewidth=3)
plt.legend(loc="best")


# In[ ]:



