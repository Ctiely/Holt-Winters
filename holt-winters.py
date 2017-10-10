# -*- coding: utf-8 -*-
"""

@author: Liu Yang, Renmin University of China
@Email:  814868906@qq.com

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import itertools
from sklearn import linear_model


class HoltWinters(object):
    
    """
    基于网格搜寻最优值的Holt-Winters算法,借鉴github作者etlundquist编写的HoltWinters算法,
    加之自己的改进编写而成。
    网址:https://github.com/etlundquist/holtwint
    该类以数据挖掘思想作为背景,将时间序列划分为训练集和测试集,目的是为了寻找到合适的参数alpha,
    beta和gamma。
    算法以MAPE作为评价标准,使用者也可以自定义评价指标,对应修改类方法Compute_Mape即可,
    方法GridSearch也需要进行修改。
    @params:
        - ts:            时间序列(序列时间由远及近)
        - p[int]:        时间序列的周期
        - test_num[int]: 测试集长度
        - sp[int]:       计算初始化参事所需要的周期数(周期数必须大于1)
        - ahead[int]:    需要预测的滞后数
        - mtype[string]: 时间序列方法:累加法或累乘法 ['additive'/'multiplicative']
    """
    
    def __init__(self, ts, p, test_num, sp=2, mtype='additive'):
        self.ts = ts
        self.p = p
        if test_num <= 0:
            raise IOError("At least a test data are required!")
        self.test_num = test_num
        self.ts_train = ts[:-test_num]
        self.ts_test = ts[-test_num:]
        self.mtype = mtype
        if self.mtype not in ['additive', 'multiplicative']:
            raise IOError("Parameter mtype accept only 'additive' and 'multiplicative'!")
        if sp < 2:
            raise IOError("At least two cycles of data are required!")
        self.sp = sp
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 0.1
        self.pred = None
        self.a_ = None
        self.b_ = None
        self.s_ = None
        
    def Compute_InitValues(self):
        p = self.p
        sp = self.sp
        initSeries = pd.Series(self.ts_train[:p*sp])
    
        if self.mtype == 'additive':
            rawSeason  = initSeries - initSeries.rolling(365, 365, center=True).mean()
            initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
            initSeason = pd.Series(initSeason) - np.mean(initSeason)
            deSeasoned = [initSeries[v] - initSeason[v % p] for v in range(len(initSeries))]
        else:
            rawSeason  = initSeries / initSeries.rolling(365, 365, center=True).mean()
            initSeason = [np.nanmean(rawSeason[i::p]) for i in range(p)]
            initSeason = pd.Series(initSeason) / math.pow(np.prod(np.array(initSeason)), 1/p)
            deSeasoned = [initSeries[v] / initSeason[v % p] for v in range(len(initSeries))]
            
        lm = linear_model.LinearRegression()
        lm.fit(pd.DataFrame({'time': [t+1 for t in range(len(initSeries))]}), pd.Series(deSeasoned))
        self.a_, self.b_, self.s_ = float(lm.intercept_), float(lm.coef_), list(initSeason)
    
    @classmethod
    def check_parameter(cls, alpha, beta, gamma):
        if alpha < 0 or alpha > 1:
            raise Exception("Parameter alpha must be in [0, 1]!")
        if beta < 0 or beta> 1:
            raise Exception("Parameter beta must be in [0, 1]!")
        if gamma < 0 or gamma > 1:
            raise Exception("Parameter gamma must be in [0, 1]!")
        
    def Compute_ExpSmooth(self, alpha, beta, gamma):
        """
        @params:
            - alpha[float]:  user-specified level  forgetting factor
            - beta [float]:  user-specified slope  forgetting factor
            - gamma[float]:  user-specified season forgetting factor
        """
        HoltWinters.check_parameter(alpha, beta, gamma)
        
        smoothed = []
        Lt1, Tt1, St1 = self.a_, self.b_, self.s_[:]
        ts = self.ts_train
        p = self.p
        
        for t in range(len(ts)):
    
            if self.mtype == 'additive':
                Lt = alpha * (ts[t] - St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
                Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
                St = gamma * (ts[t] - Lt)         + (1 - gamma) * (St1[t % p])
                smoothed.append(Lt1 + Tt1 + St1[t % p])
            else:
                Lt = alpha * (ts[t] / St1[t % p]) + (1 - alpha) * (Lt1 + Tt1)
                Tt = beta  * (Lt - Lt1)           + (1 - beta)  * (Tt1)
                St = gamma * (ts[t] / Lt)         + (1 - gamma) * (St1[t % p])
                smoothed.append((Lt1 + Tt1) * St1[t % p])
    
            Lt1, Tt1, St1[t % p] = Lt, Tt, St
        
        self.a, self.b, self.s = Lt1, Tt1, St1
        self.smoothed = smoothed[:]
    
    def PredictValues(self, ahead):
        """
        @params:
        - ahead[int]:    预测值预测时长
        """
        Lt, Tt, St = self.a, self.b, self.s
        if self.mtype == 'additive':
            pred = [Lt + (t+1)*Tt + St[t % self.p] for t in range(ahead)]
            self.pred = pred
        else:
            pred = [(Lt + (t+1)*Tt) * St[t % self.p] for t in range(ahead)]
            self.pred = pred
        return pred
    
    @staticmethod
    def Compute_Mape(y_true, y_pred):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred)
        return(np.mean(np.abs(y_true - y_pred)/y_true) * 100)
    
    @property
    def Mape(self):
        if self.pred is None:
            raise Exception("You must run function PredictValues before" +
                            "you want to compute mape of test set!")
        y_true = self.ts_test
        y_pred = self.pred
        mape = HoltWinters.Compute_Mape(y_true, y_pred)
        self.mape = mape
        return(mape)
        
    def fit(self, alpha=None, beta=None, gamma=None):
        """
        @params:
            - alpha[float]:  user-specified level  forgetting factor (0.1 if None)
            - beta[float]:   user-specified slope  forgetting factor (0.1 if None)
            - gamma[float]:  user-specified season forgetting factor (0.1 if None)
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        
        if self.a_ is None or self.b_ is None or self.s_ is None:
            self.Compute_InitValues()
            
        self.Compute_ExpSmooth(self.alpha, self.beta, self.gamma)
        
    def GridSearch(self, alphas=None, betas=None, gammas=None):
        """
        @params:
            - alphas[list or numpy]:  user-specified level  forgetting factor 
                                      (np.arange(0.1, 1, 0.1) if None)
            - betas [list or numpy]:  user-specified slope  forgetting factor 
                                      (np.arange(0.1, 1, 0.1) if None)
            - gammas[list or numpy]:  user-specified season forgetting factor 
                                      (np.arange(0.1, 1, 0.1) if None)
        """
        if alphas is None or len(alphas) == 0:
            self.alphas = np.arange(0.1, 1, 0.1)
        if betas is None or len(betas) == 0:
            self.betas = np.arange(0.1, 1, 0.1)
        if gammas is None or len(gammas) == 0:
            self.gammas = np.arange(0.1, 1, 0.1)
            
        grids = itertools.product(self.alphas, self.betas, self.gammas)
        
        self.best_alpha, self.best_beta, self.best_gamma = 0, 0, 0
        self.best_mape = np.inf
        
        for alpha, beta, gamma in grids:
            self.fit(alpha, beta, gamma)
            y_pred = self.PredictValues(self.test_num)
            mape = HoltWinters.Compute_Mape(self.ts_test, y_pred)
            if mape < self.best_mape:
                self.best_alpha, self.best_beta, self.best_gamma = alpha, beta, gamma
                self.best_mape = mape
                self.best_pred = y_pred
        self.fit(self.best_alpha, self.best_beta, self.best_gamma)
