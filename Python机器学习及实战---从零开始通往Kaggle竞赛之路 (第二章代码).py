#!/usr/bin/python2.7
#-*- coding:utf-8 -*-
__author__ = 'wangdehua'
# https://github.com/qiwsir/algorithm/blob/master/bin_search.md

# 第二章

#################################################################### 有监督学习
#################################################################### 分类学习
# 代码 13: 肿瘤预测
# 导入pandas, numpy 
import pandas as pd 
import numpy as np 

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 
                'Bare Nuclei', 'Bland Chromation', 'Normal Nucleoli', 'Mitoses', 'Class']

# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=column_names)
data = pd.read_csv('C:/Users/lenovo/Desktop/breast-cancer.txt', names=column_names)
# 替换缺失值
data = data.replace(to_replace='?', value=np.nan)
# 删除缺失值
data = data.dropna(how='any')
print data.shape


# 代码 14: 准备训练集和测试集
# 使用 sklearn.cross_valiation 里的 train_test_split 模块分割数据
# from sklearn.cross_valiation import train_test_split
from sklearn.model_selection import train_test_split
# 随机采样 25% 用于测试
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size = 0.25, random_state = 33)

# 查验训练样本的数量和类别分布
print y_train.value_counts()
print y_test.value_counts()


# -----------------------------------------------------------------------------------
# 代码 15: 使用 线性回归模型 和 SGD 从事良性和恶性肿瘤的预测任务
# 从 sklearn.preprocessing 里导入 StandardScaler
from sklearn.preprocessing import StandardScaler
# 从 sklearn.linear_model 导入 LogisticRegression & SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# 标准化数据, 保证每个维度的特征数据方差为1, 均值为0, 使得预测结果不会被某些纬度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化 LogisticRegression &  SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier(max_iter=50)
# 调用 LogisticRegression 中 fit 函数 训练模型参数
lr.fit(X_train, y_train)
# 使用训练好的模型 lr 对 X_test 进行预测, 结果存储在变量 lr_y_predict 中
lr_y_predict = lr.predict(X_test)

# 调用 SGDClassifier 中 fit 函数训练模型参数
sgdc.fit(X_train, y_train)
# 使用训练好的模型 lr 对 X_test 进行预测, 结果存储在变量 lr_y_predict 中
sgdc_y_predict = sgdc.predict(X_test)


#代码 16: 使用线性分类模型从事良、恶肿瘤预测任务的性能
# 从 sklearn.metrics 里导入 classification_report
from sklearn.metrics import classification_report
# 使用逻辑回归自带的评分函数 score 获得模型的准确性结果
print 'Accuracy of LR Classifier: ', lr.score(X_test, y_test)
# 利用 classification_report 获得 LogisticRegression 其他三个指标的结果
print classification_report(y_test, lr_y_predict, target_names = ['Benign', 'Malignant'])

# 使用梯度下降模型自带的评分模型 score 
print 'Accuracy of SGD Classifier: ', sgdc.score(X_test, y_test)
# 利用 classification_report 获得 LogisticRegression 其他三个指标的结果
print classification_report(y_test, lr_y_predict, target_names = ['Benign', 'Malignant'])


# -----------------------------------------------------------------------------------
# Support Vector Machine
# 代码 17: 手写字体数据读取代码样例
# 从 sklearn.datasets 中导入手写数字加载器
from sklearn.datasets import load_digits
# 从通过数据加载器获得手写数字的数码图像数据存储在 digits 中
digits = load_digits()
# 检验数据规模和特征维度
print digits.data.shape


# 代码 18: 手写体数据样本分割
# 依照惯例, 在没有提供测试集的情况下, 分割数据集
# 从 sklearn.cross_valiation 导入 train_test_split 用于数据分割
# from sklearn.cross_valiation import train_test_split
from sklearn.model_selection import train_test_split
# 随机选择 75% 作为训练样本, 25%作为测试数据
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 33)

# 检测数据集大小
print y_train.shape
print y_test.shape


# 代码 19: 使用支持向量机对手写体图像进行识别
# 从sklearn.preprocessing 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从 sklearn.svm 中导入基于线性假设的支持向量机分类器 LinearSVC
from sklearn.svm import LinearSVC

# 对数据进行标准化处理
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化线性假设的支持向量机分类器 LinearSVC
lsvc = LinearSVC()
# 进行模型训练
lsvc.fit(X_train, y_train)
# 使用训练好的模型对 X_test 进行预测, 结果存储在变量 y_predict 中
y_predict = lsvc.predict(X_test)


# 代码 20: 支持向量机模型对手写数码图像识别能力的评估
# 使用模型自带的评估模型函数进行准确性评估
print 'The Accuracy of Linear SVC is: ', lsvc.score(X_test, y_test)
# 依然使用 sklearn.metrics 中的 classification_report 模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))

# 这里指出一点: 召回率, 准确率和F1指标适用于 2 分类; 但在本例中, 目标是 10 个类别. 通常做法是: 逐一评估某个类别的这三个性能指标,
# 我们把所有其他的类别看做负样本, 这样一来就创造了 10 个二分类任务.


# -----------------------------------------------------------------------------------
# Navie Bayes 朴素贝叶斯
# 朴素贝叶斯会单独考量每一维度特征被分类的条件概率, 进而综合这些概率对其所在的特征向量做出分类预测, 各个维度的特征被分类的条件概率之间是相互独立的.
# 使用 20 类新闻文本进行实验数据

# 代码 21: 读取 20 类新闻文本的数据
# 从 sklearn.datasets 导入新闻数据抓取器 fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
# fetch_20newsgroups 需要从网上下载数据
news = fetch_20newsgroups(subset='all')
# 检查数据规模与细节
print len(news.data)
print news.data[0]


# 代码 22: 20 类新闻文本数据分类
# 从 sklearn.model_selection 导入 train_test_split
from sklearn.model_selection import train_test_split
# 随机采样 25%
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size = 0.25, random_state = 33)


# 代码23: 使用朴素贝叶斯分类器对新闻文本数据进行类别预测
# 从 sklearn.feature_extraction.text 导入文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 从 sklearn.naive_bayes 导入 朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行评估
mnb.fit(X_train, y_train)
# 对测试数据样本进行类别预测, 结果存在 y_predict 中
y_predict = mnb.predict(X_test)


# 代码 24: 对朴素贝叶斯分类器在新闻文本数据上的表现进行评估
# 从 sklearn.metrics 导入 classification_report 进行详细的分类性能报告
from sklearn.metrics import classification_report
print 'The accuracy of Navie Bayes Classifier is: ', mnb.score(X_test, y_test)
print classification_report(y_test, y_predict, target_names = news.target_names)


# -----------------------------------------------------------------------------------
# K-neighbor
# 代码 25: 读取 iris 数据集
# 从 sklearn.datasets 导入 iris 加载器
from sklearn.datasets import load_iris
# 使用加载器读取数据并存入变量 iris
iris = load_iris()
print iris.data.shape
# 查看数据说明
print iris.DESCR


# 代码 26: 对 iris 数据随机分割
# 从 sklearn.model_selection  导入 train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 33)


# 代码 27: 使用 K-neighbor 对数据进行预测
# 从 sklearn.preprocessing 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从 sklearn.neighbors 选择导入 KNeighborClassifier
from  sklearn.neighbors import KNeighborsClassifier

# 对训练集及测试集的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 使用 K-neighbors classifier 对数据尽心分类预测
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)


# 代码 28: 对 K-neighbors 分类在 iris 数据上的预测性能进行评估
# 使用模型自带的评估函数进行准确性测评
print 'The accuracy of K-neighbors Classifier is: ', knc.score(X_test, y_test)
# 依然使用 sklearn.metrics 中的 classification_report 模块对预测结果进行详细的分析
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names = iris.target_names)


# -----------------------------------------------------------------------------------
# 决策树
# 常用的节点分裂度量方式: 信息熵, 基尼不纯度
# 代码 29: 泰坦尼克号数据
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# print titanic.head()
# print titanic.info()

# 代码 30: 使用决策树预测泰坦尼克号乘客的生还情况
# 注意: 特征选择
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 对当前选择的特征进行探查
# print X.info()
# 'age' 数据只有 633 个, 需要填充;
# 'sex' & 'pclass' 是类别变量, 转化为 0/1 形式
X['age'].fillna(X['age'].mean(), inplace=True)

# 使用 sklearn-learn.feature_extraction 中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

# 转化特征后, 发现类别变量单独剥离出来
X = vec.fit_transform(X.to_dict(orient = 'record'))
print vec.feature_names_

# 数据分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)

# 导入 sklearn.tree 决策树分类器
from sklearn.tree import DecisionTreeClassifier

# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()

# 使用分割的训练数据进行训练模型
dtc.fit(X_train, y_train)

# 使用训练好的模型对特征数据进行预测
y_predict = dtc.predict(X_test)


# 代码 31: 决策树模型对泰坦尼克号的乘客是否生还的预测性能
# 导入 sklearn.metrics classification_report
from sklearn.metrics import classification_report
# 输出预测准确性
print dtc.score(X_test, y_test)
# 输出更详细的分类性能
print classification_report(y_predict, y_test, target_names = ['died', 'survived'])


# -----------------------------------------------------------------------------------
# 集成模型
# 代码 32: 集成模型对泰坦尼克号乘客是否生还的预测
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

X['age'].fillna(X['age'].mean(), inplace=True)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X.to_dict(orient = 'record'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)

# 使用单一决策树训练及预测
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_predict = dtc.predict(X_test)

# 使用随机森林训练模型及预测
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)

# 使用梯度提升树训练模型及预测
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_predict = gbc.predict(X_test)


# 代码 33: 集成模型对泰坦尼克号乘客是否生还的预测性能
from sklearn.metrics import classification_report

# 单一决策树
print 'The accuracy of decision tree is: ', dtc.score(X_test, y_test)
print classification_report(dtc_y_predict, y_test)

# 随机森林
print 'The accuracy of random forest classifier is: ', rfc.score(X_test, y_test)
print classification_report(rfc_y_predict, y_test)

# 梯度提升决策树
print 'The accuracy of gradient tree boosting is: ', gbc.score(X_test, y_test)
print classification_report(gbc_y_predict, y_test)


#################################################################### 回归预测
# 线性回归器, 优化目标: argmin_{w, b} L(w, b)=argmin_{w, b}sum_{k=1}^{m}(f(w, x, b) - y^{k})^2, 使用梯度下降法
# 代码 34: 美国波士顿房价预测
from sklearn.datasets import load_boston
boston = load_boston()

# 输出数据描述
# print boston.DESCR


# 代码 35: 房价数据分割
from sklearn.model_selection import train_test_split
import numpy as np
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33) # X(变量), y(响应变量)

# 分析回归目标值的差异
print 'The max target value is: ', np.max(boston.target)
print 'The min target value is: ', np.min(boston.target)
print 'The average target value is: ', np.mean(boston.target)


# 训练与测试数据标准化
# 从 sklearn.preprocessing 导入数据标准化模块
from sklearn.preprocessing import StandardScaler

# 分别初始化特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练数据和测试的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))  # 参考: http://blog.csdn.net/llx1026/article/details/77940880
y_test = ss_y.transform(y_test.reshape(-1, 1))


# 代码 37: 使用线性回归模型 LinearRegression & SGDRegressor 分别对美国波士顿地区房价进行预测
# 从 sklearn.linear_model 导入 LinearRegression
from sklearn.linear_model import LinearRegression
# 使用默认配置初始化线性回归器 LinearRegression
lr = LinearRegression()
# 使用训练数据进行参数估计
lr.fit(X_train, y_train)
# 对测试数据进行回归预测
lr_y_predict = lr.predict(X_test)

# 从 sklearn.linear_model 导入 SGDRegressor
from sklearn.linear_model import SGDRegressor
# 使用默认配置初始化线性回归器 SGDRegressor
sgdr = SGDRegressor(max_iter=5)
# 使用训练数据进行参数估计
sgdr.fit(X_train, y_train.ravel())
# 对测试数据进行回归预测
sgdr_y_predict = sgdr.predict(X_test)


# 代码 38: 使用回归评价机制及 R-squared 对模型的回归机制做评价
# MSE(平均绝对误差), MSE(均方误差)
# 使用 LinearRegression 模型自带的评估模型
print 'The value of default measurement of LinearRegression is: ', lr.score(X_test, y_test)

# 使用 sklearn.metrics 中的 r2_score, mean_squared_error, mean_absoluate_error 用于回归性能的评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print 'The value of R-squared of LinearRegression is: ', r2_score(y_test, lr_y_predict)
print 'The mean squared error of LinearRegression is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict))
print 'The mean absoluate error of LinearRegression is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict))

print 'The value of default of SDGRegressor is', sgdr.score(X_test, y_test)
print 'The vlue of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict)
print 'The mean squared error of SGDRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))
print 'The mean absolute error of SGDRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))


# -----------------------------------------------------------------------------------
# 支持向量机(回归)
# 代码 39: 使用三种不同的核函数配置的支持向量机回归模型进行训练, 并且分别对测试数据做预测
# 从 sklearn.svm 中导入支持向量机模型
from sklearn.svm import SVR

# 使用线性核函数
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train.ravel())
linear_svr_y_predict = linear_svr.predict(X_test)

# 使用多项式核函数
poly_svr = SVR(kernel="poly")
poly_svr.fit(X_train, y_train.ravel())
poly_svr_y_predict = poly_svr.predict(X_test)

# 使用径向基核函数
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train.ravel())
rbf_svr_y_predict = rbf_svr.predict(X_test)


# 代码 40: 对三种核函数配置下的支持向量机回归模型在相同测试集上进行性能评估
# 使用 R-squared, MSE & MAE 指标对三种配置的支持向量机模型进行性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print 'R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test)
print 'The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))
print 'The mean absolute error of linear SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))

print 'R-squared value of poly SVR is: ', poly_svr.score(X_test, y_test)
print 'The mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))
print 'The mean absolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))

print 'R-squared value of RBF SVR is: ', rbf_svr.score(X_test, y_test)
print 'The mean squared error of RBF SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))
print 'The mean absolute error of RBF SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))
# 可以使用其他核函数(核函数是一种非常有用的特征映射技巧, 将原来的特征映射到更高维的空间, 从而达到新的高维度特征线性可分的程度)来改变模型性能


# -----------------------------------------------------------------------------------
# K-nearest-neighborss (回归)
# 代码 41: 使用两种(K 个目标数值使用普通的算术平均算法, 距离差异进行加权平均)不同的配置的 K 近邻回归模型对美国波士顿房价进行预测
# 从 sklearn.neighbors 导入 KNeighborRegressor (K 近邻回归器)
from sklearn.neighbors import KNeighborsRegressor

# 初始化 K 近邻回归器, 并调整配置, 使得预测的方式为平均回归: weights='uniform'
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# 初始化 K 近邻回归器, 并调整配置, 使得预测的方式为距离加权回归: weights='distance'
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)


# 代码 42: 性能评估
# 使用 R-squared, MSE & MAE 三种指标对平均回归配置的 K 近邻模型在测试集上进行性能评估
print 'R-squared value of uniform-weighted KNeighborRegression: ', uni_knr.score(X_test, y_test)
print 'The mean squared error of uniform-weighted KNeighborRegression: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict))
print 'The mean absolute error of uniform-weighted KNeighborRegression: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict))

# 使用 R-squared, MSE & MAE 三种指标对距离加权回归配置的 K 近邻模型在测试集上进行性能评估
print 'R-squared value of distance-weighted KNeighborRegression: ', dis_knr.score(X_test, y_test)
print 'The mean squared error of distance-weighted KNeighborRegression: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict))
print 'The mean absolute error of distance-weighted KNeighborRegression: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict))


# -----------------------------------------------------------------------------------
# 回归树
# 回归树在选择不同特征作为分裂节点的策略上, 与决策树类似. 不同之处在于回归树节点的数据类型不是离散值, 而是连续型数据. 
# 决策树每个叶节点依照训练数据表现的概率倾向决定了其最终的预测类别; 而回归树的叶节点确是一个个具体的值, 叶节点返回的是“一团”训练数据的均值, 而不是具体的/连续的预测值.
# 代码 43: 使用回归树对波士顿房价进行训练及预测
# 从 sklearn.tree 导入 DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr_y_predict = dtr.predict(X_test)


# 代码 44: 单一回归树的性能评估
# 使用 R-squared, MSE & MAE 三种指标对默认配置的回归树在测试集上进行性能评估
print 'R-squared value of DecisionTreeRegressor: ', dtr.score(X_test, y_test)
print 'The mean squared error of DecisionTreeRegressor: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict))
print 'The mean absolute error of DecisionTreeRegressor: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict))


# -----------------------------------------------------------------------------------
# 集成模型
# 补充一种集成方法: 极端随机森林(Extremely Randomized Trees), 与普通的随机森林模型不同的是: 极端森林在每当构建一棵树的分裂节点的时候, 不会任意的选择特征, \n
# 而是先随机收集一部分特征, 然后利用信息熵或基尼不纯度等指标挑选最佳的节点特征
# 代码 45: 使用三种集成模型对美国波士顿房价进行训练及预测
# 从 sklearn.ensemble 中导入 RandomForestRegressor, ExtraTreeGressor & GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train.ravel())
rfr_y_predict = rfr.predict(X_test)

# ExtraTreesRegressor
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train.ravel())
etr_y_predict = etr.predict(X_test)

# GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train.ravel())
gbr_y_predict = gbr.predict(X_test)


# 代码 46: 性能评估
# 使用 R-squared, MSE & MAE 三种指标对 RandomForestRegressor 在测试集上进行性能评估
print 'R-squared value of RandomForestRegressor: ', rfr.score(X_test, y_test)
print 'The mean squared error of RandomForestRegressor: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict))
print 'The mean absolute error of RandomForestRegressor: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict))

# 使用 R-squared, MSE & MAE 三种指标对 ExtraTreesRegressor 在测试集上进行性能评估
print 'R-squared value of ExtraTreesRegressor: ', etr.score(X_test, y_test)
print 'The mean squared error of ExtraTreesRegressor: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict))
print 'The mean absolute error of ExtraTreesRegressor: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict))
# 查看特征贡献度
print np.sort(zip(etr.feature_importances_, boston.feature_names), axis=0)

# 使用 R-squared, MSE & MAE 三种指标对 GradientBoostingRegressor 在测试集上进行性能评估
print 'R-squared value of GradientBoostingRegressor: ', gbr.score(X_test, y_test)
print 'The mean squared error of GradientBoostingRegressor: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict))
print 'The mean absolute error of GradientBoostingRegressor: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict))



#################################################################### 无监督学习
# -----------------------------------------------------------------------------------
# 数据聚类
# K-means
# 代码 47: K-means 算法在手写数字图像数据上的使用示例
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 使用 pandas 读取数据
# digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
# digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)
digits_train = pd.read_csv('C:/Users/lenovo/Desktop/optdigits.tra', header=None)
digits_test = pd.read_csv('C:/Users/lenovo/Desktop/optdigits.tes', header=None)

# 从训练集和测试数据集上分离出 64 维像素特征和 1 维数字目标
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 从 sklearn.cluster 导入 KMeans 模型
from sklearn.cluster import KMeans
# 初始化 KMeans 模型, 并设置聚类中心数量为 10.
kmeans = KMeans(n_clusters = 10)
kmeans.fit(X_train)
# 逐条判断每个测试图像所属的聚类中心
y_pred = kmeans.predict(X_test)


# 代码 48: 使用 ARI 进行 K-means 聚类性能评估
# 从 sklearn 导入度量函数库 metrics
from sklearn import metrics
# 使用 ARI 进行 KMeans 聚类性能评估
print metrics.adjusted_rand_score(y_test, y_pred)