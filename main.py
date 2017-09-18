# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from ganzhiqi import Perceptron
from matplotlib_listencolormap import plot_decision_regions

#设置中文字体，防止中文乱码
plt.rcParams['font.sans-serif']=['SimHei']

df = pd.read_csv('test.csv', header=None)

#读取前100个样本的第4列
y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1,1)

x = df.loc[0:100, [0,2]].values

"""
使用感知器算法训练一个神经网络
"""
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x,y)

"""
使用训练好的神经网络预测分类单元
"""
plot_decision_regions(x,y,ppn,resolution=0.02)
plt.xlabel(u'花径长度')
plt.ylabel(u'花瓣长度')
plt.legend(loc='upper left')
plt.show()
