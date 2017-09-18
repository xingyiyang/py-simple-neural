# -*- coding: utf-8 -*-

import numpy as np

class Perceptron(object):
    """
    eta: 学习率,(0,1),一般是使用者自定义，然后根据经验调整
    n_iter: 权重向量的训练次数，这里只训练10次
    w_: 神经分叉权重向量
    errors_: 用于记录神经元判断错误的次数
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, x, y):
        """
        输入训练数据，训练神经元
        x： 输入样本的向量
        y： 样本对应的分类
        x: shape[n_sample, n_features]: 样本数量，每个样本的特征维度
        x:[[1,2,3],[4,5,6]]，有2个样本，每个样本有3个特征
        """
        #初始化权重向量的每个分量，这里初始化的值为0
        #+1是为了w0，权重向量W的维度加1，多出的那一维用于计算激活函数的阈值
        self.w_ = np.zeros(1+x.shape[1])
        self.errors_ = []
        
        """
        开始训练
        每次出现预测错误的时候，把样本重新输入神经元，更新权重向量
        这里设置的迭代次数我n_iter=10
        如果迭代10次都无法分类正确，则终止
        """
        for _ in range(self.n_iter):
            errors = 0
            """
            x:[[1,2,3],[4,5,6]]
            y:[1,-1]
            zip(x,y): [([1, 2, 3], 1), ([4, 5, 6], -1)]
            """
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                
                #xi是向量，target是实数
                #更新权值
                self.w_[1:] += update * xi
                
                #更新阈值
                self.w_[0] += update;
                
                """
                统计预测的错误次数
                int(True)=1
                int(False)=0
                """
                errors += int(update !=0.0)
                self.errors_.append(errors)
                
    def net_input(self, x):
        """
        输入向量与权重向量做点积求和
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]
            
    def predict(self,x):
        """
        激活函数得到输出为1或-1的分类
        大于等于0输出为1
        小于0输出为-1
        """
        return np.where(self.net_input(x) >= 0.0 , 1,-1)