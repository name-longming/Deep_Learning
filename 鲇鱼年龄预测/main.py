# 读取文件
import numpy as np
import os
import matplotlib.pyplot as plt

data_X = []
data_Y = []
# 将性别（M：雄性，F：雌性，I：未成年）映射成数字
sex_map = { 'I': 0, 'M': 1, 'F': 2 }
with open ('work/AbaloneAgePrediction.txt') as f:
    for line in f.readlines():
        line = line.split(',')
        line[0] = sex_map[line[0]]
        data_X.append(line[:-1])
        data_Y.append(line[-1:])
# 转换为nparray
data_X = np.array(data_X, dtype='float32')
data_Y = np.array(data_Y, dtype='float32')

data_X  = data_X.reshape(data_X.shape[0],-1).T
#将测试集的维度降低并转置。
data_Y = data_Y.reshape(data_Y.shape[0], -1).T

# 检查大小
print('data shape', data_X.shape, data_Y.shape)
print('data_x shape[1]', data_X.shape[1])
# 归一化
for i in range(data_X.shape[0]):
    _min = np.min(data_X[i,:])                            #每一列的最小值
    _max = np.max(data_X[i,:])                            #每一列的最大值
    data_X[i,:] = (data_X[i,:] - _min) / (_max - _min)  #归一化到0-1之间

def initialize_with_zeros(dim):
    """
        此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。

        参数：
            dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）

        返回：
            w  - 维度为（dim，1）的初始化向量。
            b  - 初始化的标量（对应于偏差）
    """
    w = np.zeros(shape = (dim,1))
    b = 0
    #使用断言来确保我要的数据是正确的
    assert(w.shape == (dim, 1)) #w的维度是(dim,1)
    assert(isinstance(b, float) or isinstance(b, int)) #b的类型是float或者是int

    return (w , b)

def propagate(w, b, X, Y):
    """
    实现前向和后向传播的成本函数及其梯度。
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 矩阵类型为（num_px * num_px * 3，训练数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

    返回：
        cost- 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w相同的形状
        db  - 相对于b的损失梯度，因此与b的形状相同
    """
    m = X.shape[1]

    #正向传播
    A = np.dot(w.T,X) + b #计算激活值，请参考公式2。

    error = A - Y
    cost = error * error
    cost = np.sum(cost) / m

    #反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T) #请参考视频中的偏导公式。
    db = (1 / m) * np.sum(A - Y) #请参考视频中的偏导公式。

    #使用断言确保我的数据是正确的
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    #创建一个字典，把dw和db保存起来。
    grads = {
                "dw": dw,
                "db": db
             }
    return (grads , cost)


def optimize(w , b , X , Y , num_iterations , learning_rate , print_cost = False):
    """
    此函数通过运行梯度下降算法来优化w和b

    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值

    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。

    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        #记录成本
        if i % 100 == 0:
            costs.append(cost)
        #打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i,cost))

    params  = {
                "w" : w,
                "b" : b }
    grads = {
            "dw": dw,
            "db": db }
    return (params , grads , costs)

w, b = initialize_with_zeros(data_X.shape[0])

parameters, grads, costs = optimize(w, b, data_X, data_Y, num_iterations = 10000 , learning_rate = 0.5, print_cost = True)

w , b = parameters["w"] , parameters["b"]
X = np.array([1,0.44,0.365,0.125,0.516,0.2155,0.114,0.155])
X = np.reshape(X, [8, 1])
for i in range(data_X.shape[0]):
    data_X[i, :] = (data_X[i, :] - _min) / (_max - _min)  #归一化到0-1之间

A = np.dot(w.T , X) + b
print(A)
