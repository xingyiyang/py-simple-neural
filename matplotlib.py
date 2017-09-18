# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#设置中文字体，防止中文乱码
plt.rcParams['font.sans-serif']=['SimHei']

df = pd.read_csv('test.csv', header=None)

#读取前100个样本的第4列
y = df.loc[0:100, 4].values
#把第4列的字符串描述转换成1或-1，量化
y = np.where(y == 'Iris-setosa', -1,1)

#把第0列和第2列取出来
x = df.iloc[0:100, [0,2]].values

#scatter用于画散点图
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')

#u是为了防止中文乱码
plt.xlabel(u'花瓣长度')
plt.ylabel(u'花径长度')
plt.legend(loc='upper left')
plt.show()