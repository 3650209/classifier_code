import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

learning_rate = 2e-2
epochs = 10000

# batch_size=100

def data_preprocessing(data):
    # print("Missing values", data.isnull().sum())
    data.fillna(0, inplace=True)

    xy_numpy = data.to_numpy()

    x = xy_numpy[:, 2:]  # x为2038*53
    y = xy_numpy[:, 1].reshape(-1, 1)  # y为2038*1

    print(x.data.shape)
    print(y.data.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    X_train = torch.Tensor(torch.from_numpy(X_train))
    y_train = torch.Tensor(torch.from_numpy(y_train))
    X_test = torch.Tensor(torch.from_numpy(X_test))
    y_test = torch.Tensor(torch.from_numpy(y_test))

    y_train = torch.tensor(y_train.squeeze(1), dtype=torch.long) - 1
    y_test = torch.tensor(y_test.squeeze(1), dtype=torch.long) - 1

    print(y_train)
    return X_train, X_test, y_train, y_test


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义多层神经网络
        self.fc1 = torch.nn.Linear(53, 25)
        self.fc11 = torch.nn.Linear(25, 15)
        self.fc12 = torch.nn.Linear(15, 15)
        self.fc2 = torch.nn.Linear(15, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc12(self.fc11(x))
        x = F.dropout(x, p=0.1)  # dropout 1
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1)  # dropout 2
        y_pred = torch.nn.functional.softmax(x)

        return y_pred


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data = torch.randn(m.weight.data.size()[0], m.weight.data.size()[1])
        m.bias.data = torch.randn(m.bias.data.size()[0])
        nn.init.kaiming_normal_(m.weight.data)


xy = pd.read_csv('data1.csv', delimiter=',', encoding='gbk', dtype=np.float32)

X_train, X_test, y_train, y_test = data_preprocessing(xy)

model = Model()
model.apply(weight_init)

# 定义损失函数及优化器
criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练
Loss = []
for epoch in range(epochs):
    y_pred = model(X_train)
    # print(y_pred)
    # 计算误差
    loss = criterion(y_pred, y_train)

    # 每迭代100次打印Lost并记录
    if epoch % 100 == 0:
        print('[<<%d/%5d>>] loss: %.3f' %
              (epoch + 1, epochs, loss.item()))
        Loss.append(loss.item())

    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新梯度
    optimizer.step()

# 绘制loss曲线
plt.plot(range(0, epochs, 100), Loss)
plt.xlabel('epoch')
plt.ylabel('finalosses')
plt.show()

predictions = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())
        # print(y_pred.argmax().item())

# 分类准确率
score = accuracy_score(y_test, predictions)

print(score)

# 保存模型
torch.save(model, 'prediction.pt')
