#导入所需模块
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
import pandas as pd
import numpy as np
import seaborn as sns
from graphviz import Digraph
from matplotlib import pyplot as plt
import os

#将数据读入dataset
dataset=pd.read_csv('data.csv')
#dataset.set_index('ID',inplace=True)
#dataset.isna().sum()
print(dataset.head())
#plt.style.use('seaborn')
#dataset['Result'].value_counts().plot(kind='bar')
#plt.xticks(rotation=0)
#plt.show()
#plt.style.use('seaborn')
#sns.pairplot(data=dataset,hue='Result')
#plt.show()
sns.set()
ox=dataset['Temp']
sns.displot(ox,kde=True,bins=15)
plt.show()