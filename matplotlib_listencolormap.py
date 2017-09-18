# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from ganzhiqi import Perceptron

"""
使用训练好的神经网络预测分类单元
"""
def plot_decision_regions(x,y,classifier,resolution=0.02):
    markers = ('s','x','o','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    """
    花径和花瓣的最大最小值
    """
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()
    
    #print(x1_min, x1_max)
    #print(x2_min, x2_max)
    
    """
    meshgrid：把两个数组相互作用后转化成两个矩阵
    meshgrid（x,y）
    矩阵1： 把x数组作为一行，重复这些行数，总共y.length行
    矩阵2： 把y数组作为一行，重复这些行数，总共X.length行
    """
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                        np.arange(x2_min, x2_max, resolution))
    
    #ravel 复制并把矩阵还原成单维向量
    #模型分类后得到的结果,需要转置
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    
    z = z.reshape(xx1.shape)
    
    #绘制分界线
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    
    #xlim(x1,x2),x1是坐标的起始节点，x2是坐标的末尾节点
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y==c1, 0], y=x[y==c1, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=c1)
    