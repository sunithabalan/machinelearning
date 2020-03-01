# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('train.csv')
print(diabetes.columns)
diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))
print(diabetes.groupby('Outcome').size())
#
import seaborn as sns
sns.countplot(diabetes['Outcome'],label="Count")