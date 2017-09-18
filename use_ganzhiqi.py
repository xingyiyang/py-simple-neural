# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from ganzhiqi import Perceptron

df = pd.read_csv('test.csv', header=None)

#读取前100个样本的第4列
y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1,1)
x = df.iloc[0:100, [0,2]].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('epochs')
plt.ylabel('wrong classify ')
plt.show()
