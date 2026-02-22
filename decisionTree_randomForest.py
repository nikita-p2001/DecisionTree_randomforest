#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


# In[3]:


from sklearn.datasets import load_breast_cancer


# In[5]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[6]:


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
tree = DecisionTreeClassifier(max_depth=None, random_state=42)
tree.fit(X_train, y_train)

print("Tree Train Score:", tree.score(X_train, y_train))
print("Tree Test Score:", tree.score(X_test, y_test))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("RF Train Score:", rf.score(X_train, y_train))
print("RF Test Score:", rf.score(X_test, y_test))

