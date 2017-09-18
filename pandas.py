# -*- coding: utf-8 -*-

import pandas as pd

file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#读取csv的文件，一般第一行都是用于说明的，而这里直接是数据，所以header=None，表示直接读取
df = pd.read_csv(file, header=None)
#把数据写进文件保存在本地，index表示是否显示行名
df.to_csv('test.csv', index=False, header=None)
#显示前10个样本
print(df.head(10))