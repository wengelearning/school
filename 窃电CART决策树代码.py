# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:57:56 2017

@author: s
"""

import pandas as pd
from random import shuffle
datafile='F:/P-file/test.xls'
data=pd.read_excel(datafile)
data=data.as_matrix()
shuffle(data)
p=0.8
train=data[:int(len(data)*p),:]
test=data[int(len(data)*p):,:]

from sklearn.tree import DecisionTreeClassifier
treefile='F:/tree.pkl'
tree=DecisionTreeClassifier()
tree.fit(train[:,:3],train[:,3])
from sklearn.externals import joblib
joblib.dump(tree,treefile)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
 
 
def confusion_matrix_plot_matplotlib(y_truth, y_predict, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_truth, y_predict)
    plt.matshow(cm, cmap=cmap)  # 混淆矩阵图
    plt.colorbar()  # 颜色标签
    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.show()  # 显示作图结果
confusion_matrix_plot_matplotlib(train[:,3],tree.predict(train[:,:3]))

from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(test[:,3],tree.predict_proba(test[:,:3])[:,1],pos_label=1)
plt.plot(fpr,tpr,linewidth=2,label='ROC OF CART')
plt.xlabel('false p r')
plt.ylabel('true p r')
plt.ylim(0,1.05)
plt.xlim(0,1.05)
plt.legend(loc=4)
plt.show()