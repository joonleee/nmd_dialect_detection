#!/usr/bin/env python
# coding: utf-8

# # XGBClassifier()

# ### Import necessary packages

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ### Load MFCC data into X and y
# > Convert to dataframe

# In[2]:


X = np.load('/Users/pogo/Downloads/MFCC_features_all.npy')
y = np.load('/Users/pogo/Downloads/MFCC_label_all.npy')
X_df = pd.DataFrame(X)
# X_df[0]
y_df = y.astype(np.float)


# # Split data into train and test datasets

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.33, random_state=7)

model = XGBClassifier()

eval_set = [(X_train, y_train), (X_test, y_test)]


# # Fit the model

# In[ ]:


model.fit(X_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# # Model accuracy

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

results = model.evals_result()
epochs = len(results["validation_0"]["merror"])
x_axis = range(0, epochs)


# #### Accuracy: 74.63%
