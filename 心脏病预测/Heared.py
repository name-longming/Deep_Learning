import pandas as pd
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random

csv_file=pd.read_csv('./datasets/dataset.csv', header=1, dtype=np.float32)
csv_file = np.array(csv_file)
print(csv_file.shape)
train = csv_file

# 计算train数据集的最大值，最小值，平均值
maximums, minimums, avgs = train[:,:-1].max(axis=0), train[:,:-1].min(axis=0), \
                           train[:,:-1].sum(axis=0) / train[:,:-1].shape[0]

# 记录数据的归一化参数，在预测时对数据做归一化
global max_values
global min_values
global avg_values
max_values = maximums
min_values = minimums
avg_values = avgs

# 对数据进行归一化处理
for i in range(13):
    train[:, i] = (train[:, i] - min_values[i]) / (maximums[i] - minimums[i])

class Regressor(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()

        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc1 = Linear(in_features=13, out_features=10)
        self.fc2 = Linear(in_features=10, out_features=10)
        # 定义一层全连接输出层，输出维度是1
        self.fc3 = Linear(in_features=10, out_features=1)

    # 网络的前向计算
    def forward(self, inputs):
        outputs1 = self.fc1(inputs)
        outputs1 = paddle.tanh(outputs1)
        outputs2 = self.fc2(outputs1)
        outputs2 = paddle.tanh(outputs2)
        outputs_final = self.fc3(outputs2)
        outputs_final = F.sigmoid(outputs_final)
        return outputs_final

# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
model.train()
# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())
EPOCH_NUM = 1000  # 设置外层循环次数
BATCH_SIZE = 10  # 设置batch大小

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(train)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [train[k:k + BATCH_SIZE] for k in range(0, len(train), BATCH_SIZE)]
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])  # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1:])  # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor形式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)

        # 前向计算
        predicts = model(house_features)

        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        if iter_id % 10 == 0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

        # 反向传播
        avg_loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()

# 保存模型参数，文件名为LR_model.pdparams
paddle.save(model.state_dict(), 'LR_model.pdparams')
print("模型保存成功，模型参数保存在LR_model.pdparams中")

# 参数为保存模型参数的文件地址
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()


train_test = train[:,:-1]
label_test = train[:,-1:]

one_data = paddle.to_tensor(train_test)

predict = model(one_data)
Y_prediction = np.zeros((predict.shape[0],1))
for i in range(predict.shape[0]):
    # 将概率a [0，i]转换为实际预测p [0，i]
    Y_prediction[i, 0] = 1 if predict[i, 0] > 0.5 else 0
print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction - label_test)) * 100) ,"%")


