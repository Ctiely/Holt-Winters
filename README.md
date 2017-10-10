# Holt-Winters
基于网格搜寻最优值的Holt-Winters算法
借鉴github作者etlundquist编写的HoltWinters算法,加之自己的改进编写而成。
原作者网址:https://github.com/etlundquist/holtwint
该类以数据挖掘思想作为背景,将时间序列划分为训练集和测试集,目的是为了寻找到合适的参数alpha,beta和gamma。
算法以MAPE作为评价标准,使用者也可以自定义评价指标,对应修改类方法Compute_Mape即可,方法GridSearch也需要进行修改。
HoltWinters:
@params:
    - ts:            时间序列(序列时间由远及近)
    - p[int]:        时间序列的周期
    - test_num[int]: 测试集长度
    - sp[int]:       计算初始化参事所需要的周期数(周期数必须大于1)
    - ahead[int]:    需要预测的滞后数
    - mtype[string]: 时间序列方法:累加法或累乘法 ['additive'/'multiplicative']
