
# coding: utf-8



from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve,auc,roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('G:\\project-felix\\Dataset\\qt_dataset.csv',encoding='utf-8')


# In[3]:


df.head()


# In[4]:


df.Result.value_counts()


# In[5]:


df.Result=df.Result.astype(str).map({'Negative':0,'Positive':1})


# In[6]:


y=df.Result
y.head()


# In[7]:


x=df.drop(['Result','ID'],axis=1)
x.head()


# In[8]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.9,random_state=1)


# In[9]:


##随机森林分类器
rf1=RandomForestClassifier()    #实例化
rf2=rf1.fit(xtrain,ytrain)      #用训练数据训练模型

result=rf2.score(xtest,ytest)
result


# In[10]:


print(rf2.classes_)
print(rf2.n_classes_)


# In[11]:


print('result:%s' %rf2.predict(xtest))


# In[12]:


print('result:%s' %rf2.predict_proba(xtest)[:,:])


# In[13]:


print('result:%s' %rf2.predict_proba(xtest)[:,1])


# In[14]:


d1=np.array(pd.Series(rf2.predict_proba(xtest)[:,1]>0.5).map({False:0, True:1}))
d2=rf2.predict(xtest)
np.array_equal(d1,d2)


# In[15]:


clf=DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
scores= cross_val_score(clf,xtrain,ytrain)
print(scores.mean())

clf=RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)
scores=cross_val_score(clf,xtrain,ytrain)
print(scores.mean())


# In[16]:


rf2.get_params


# In[31]:


# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
score_lt = []

# 每隔10步建立一个随机森林，获得不同n_estimators的得分
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    score = cross_val_score(rfc,xtrain,ytrain,cv=10).mean()
    score_lt.append([i,score])
score_lt=np.array(score_lt)

max_score =np.where(score_lt==np.max(score_lt[:,1])) [0][0]
print("max_score:",score_lt[max_score])
plt.figure(figsize=[20,5])
plt.plot(score_lt[:,0],score_lt[:,1])
plt.show()


# In[34]:


score_lt = []

# 每隔10步建立一个随机森林，获得不同n_estimators的得分
for i in range(20,40):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    score = cross_val_score(rfc,xtrain,ytrain,cv=10).mean()
    score_lt.append([i,score])
score_lt=np.array(score_lt)

max_score =np.where(score_lt==np.max(score_lt[:,1])) [0][0]
print("max_score:",score_lt[max_score])
plt.figure(figsize=[10,5])
plt.ylabel('score')
plt.xlabel('n_estimators')
plt.plot(score_lt[:,0],score_lt[:,1])
plt.show()


# In[20]:


# 建立n_estimators为45的随机森林
rfc = RandomForestClassifier(n_estimators=31, random_state=90)

# 用网格搜索调整max_depth
param_grid = {'max_depth':np.arange(1,20)}
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(xtrain, ytrain)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)


# In[96]:


param_test1={'n_estimators':range(25,500,25)}
gsearch1= GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                       min_samples_leaf=20,
                                                       max_depth=8,random_state=10),
                      param_grid=param_test1,
                      scoring='roc_auc',
                      cv=5)
gsearch1.fit(xtrain,ytrain)
print(gsearch1.best_params_,gsearch1.best_score_)


# In[98]:


param_test2={'min_samples_split':range(60,200,20),'min_samples_leaf':range(10,110,10)}
gsearch2 =GridSearchCV(estimator=RandomForestClassifier(n_estimators=31,
                                                       max_depth=8,random_state=10),
                      param_grid=param_test2,
                      scoring='roc_auc',
                      cv=5)
gsearch2.fit(xtrain,ytrain)
print(gsearch2.best_params_,gsearch2.best_score_)


# param_test3={'max_depth':range(3,30,2)}
# gsearch3 =GridSearchCV(estimator=RandomForestClassifier(n_estimators=300,
#                                        min_samples_split=60,
#                                        min_samples_leaf=10,
#                                        random_state=10),
#                 param_grid=param_test3,
#                 scoring='roc_auc',
#                  cv=5)
# gsearch3.fit(xtrain,ytrain)
# print(gsearch3.best_params_,gsearch3.best_score_)

# In[20]:


param_test3={'max_depth':range(3,30,2)}
gsearch3 =GridSearchCV(estimator=RandomForestClassifier(n_estimators=50,
                                                       min_samples_split=60,
                                                        min_samples_leaf=10,
                                                        random_state=10),
                      param_grid=param_test3,
                      scoring='roc_auc',
                      cv=5)
gsearch3.fit(xtrain,ytrain)
print(gsearch3.best_params_,gsearch3.best_score_)


# In[21]:


roc_auc_score(ytest,gsearch3.best_estimator_.predict_proba(xtest)[:,1])


# In[22]:


gsearch3.best_estimator_


# In[23]:


param_test4={'criterion':['gini','entropy'],'class_weight':[None,'balanced']}
gsearch4 =GridSearchCV(estimator=RandomForestClassifier(n_estimators=50,
                                                        max_depth=5,
                                                       min_samples_split=60,
                                                        min_samples_leaf=10,
                                                        random_state=10),
                      param_grid=param_test4,
                      scoring='roc_auc',
                      cv=5)
gsearch4.fit(xtrain,ytrain)
print(gsearch4.best_params_,gsearch4.best_score_)


# In[24]:


roc_auc_score(ytest,gsearch4.best_estimator_.predict_proba(xtest)[:,1])


# In[25]:


roc_auc_score(ytest,rf2.predict_proba(xtest)[:,1])


# In[26]:


print('各feature的重要性：%s' %rf2.feature_importances_)


# In[77]:


importances=rf2.feature_importances_
feature_names=xtest.columns
std=np.std([tree.feature_importances_ for tree in rf2.estimators_],axis=0)
indices=np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(min(20,xtrain.shape[1])):
    print("%2d) %-*s %f" %(f+1,30,xtrain.columns[indices[f]],importances[indices[f]]))
plt.figure()
plt.title('Feature importances')
plt.bar(range(xtrain.shape[1]),importances[indices],color="r",yerr=std[indices],align="center")
plt.xticks(range(xtrain.shape[1]),np.array(feature_names)[indices])
plt.xlim([-1,xtrain.shape[1]])    
plt.show()


# In[88]:


predictions_validation=rf2.predict_proba(xtest)[:,1]
fpr,tpr,thresholds=roc_curve(ytest,predictions_validation)
roc_auc=auc(fpr,tpr)
plt.title('ROC Validation')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],ls="--",c=".3")
plt.xlim([0,0.4])
plt.ylim([0.9,1])
plt.ylabel('True Positivve Rate')
plt.xlabel('False Position Rate')
plt.grid(True)


# In[60]:





# In[75]:




