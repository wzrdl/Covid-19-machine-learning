
# coding: utf-8

# In[1]:


from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'InlineBackend.figure_format="retina"')
from matplotlib.font_manager import FontProperties
fonts=FontProperties(fname="/Library/Fonts/华文细黑.ttf",size=14)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
df=pd.read_csv('C:\\Users\\ASUS\\Desktop\\SVM\\qt_dataset.csv',encoding='utf-8')
df.head()
df.Result.value_counts()
df.Result=df.Result.astype(str).map({'Negative':0,'Positive':1})
y=df.Result
y.head()
x=df.drop(['Result','ID'],axis=1)
x.head()



df.info



x.info



##计算数据缺失率
df.apply(lambda x : sum(x.isnull())/len(x))


# In[5]:


from sklearn.svm import SVC,LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[6]:


##降维
scale=StandardScaler(with_mean=True,with_std=True)
xXS=scale.fit_transform(x)
x_pca=PCA(n_components=2).fit_transform(xXS)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=2)


# In[7]:


##线性核 Linear kernal
Lsvm=LinearSVC(penalty="l2",C=2.0,random_state=3)
Lsvm.fit(train_x,train_y)
pre_y=Lsvm.predict(test_x)
##测试集准确率
metrics.accuracy_score(test_y,pre_y)


# In[8]:


##非线性核 Nonlinear kernal
rbfsvm=SVC(kernel="rbf",gamma=0.5,random_state=1,degree=3)
rbfsvm.fit(train_x,train_y)
pre_y=rbfsvm.predict(test_x)
##测试集准确率
metrics.accuracy_score(test_y,pre_y)