#导入所需模块
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
import pandas as pd
import numpy as np
from graphviz import Digraph
from matplotlib import pyplot as plt
import os

#将数据读入dataset
dataset=pd.read_csv('data.csv')
col1=dataset.columns.values.tolist()
#print(col1)
col2=col1[1:4]
#print(col2)
data_x=np.array(dataset[col2])
#print(data_x)
data_y=np.array(dataset['Result'])
#print(data_y)

x_true=data_x
y_true=data_y
feature_names=col2
target_name=['Negative','Positive']

#debug
'''
print(x_true)
print(y_true)
print(feature_names)
print(target_name)
'''
'''
#划分训练集和测试集的占比
sum=0
rate=0.1
list_x=[]
list_y=[]
for i in range (1,21):
    list_x.append(rate)
    x_train,x_test,y_train,y_test=train_test_split(x_true,y_true,test_size=rate)

    print("训练集样本大小 标签大小：",x_train.shape)
    print("测试集样本大小 标签大小：",x_test.shape)

    clf=tree.DecisionTreeClassifier(criterion="gini",splitter="best",min_samples_leaf=100,max_depth=4)
    clf.fit(x_train,y_train)
    score = clf.score(x_test, y_test)
    #sum+=score
    list_y.append(score)
    print("模型测试集准确率为：", score)
    # 绘制决策树模型
    _filename="{name}.dot".format(name=i)
    clf_dot = tree.export_graphviz(clf,out_file=_filename,feature_names= feature_names,class_names= target_name,filled= True,rounded= True)
    graph = graphviz.Source(clf_dot)
    rate+=0.04
plt.title("Graph 1")
plt.xlabel("train_test_split")
plt.ylabel("the score")
plt.plot(list_x,list_y)
plt.grid(linestyle='--')
plt.show()
'''
#os.system("dot -Tpng -o 1.png 1.dot")
#for i in range(1,20):
#    _filename={name}.dot.format(name=i)
#    _outfilename={name}.png.format(name=i)
#    os.system("dot -Tpng -o {name}.png.format(name=i) {name}.dot.format(name=i)")