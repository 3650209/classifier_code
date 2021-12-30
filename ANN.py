import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df=pd.read_csv('data1.csv',encoding='gbk')
print(df.head())


class ANN_Model(nn.Module):
    def __init__(self,input_features=53,hidden1=20,hidden2=20,out_features=4):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x

def data_preprocessing(data):
    # print("Missing values", data.isnull().sum())
    data.fillna(0, inplace=True)

    xy_numpy = data.to_numpy()

    x = xy_numpy[:, 2:]  # x为2038*53
    y = xy_numpy[:, 1].reshape(-1, 1)  # y为2038*1

    print(x.data.shape)
    print(y.data.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    X_train = torch.Tensor(torch.from_numpy(X_train))
    y_train = torch.Tensor(torch.from_numpy(y_train))
    X_test = torch.Tensor(torch.from_numpy(X_test))
    y_test = torch.Tensor(torch.from_numpy(y_test))

    y_train = torch.tensor(y_train.squeeze(1), dtype=torch.long) - 1
    y_test = torch.tensor(y_test.squeeze(1), dtype=torch.long) - 1

    print(y_train)
    return X_train, X_test, y_train, y_test

xy = pd.read_csv('data1.csv', delimiter=',', encoding='gbk', dtype=np.float32)

X_train, X_test, y_train, y_test = data_preprocessing(xy)

torch.manual_seed(20)
model=ANN_Model()

print(model.parameters)

loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)


epochs=10000
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss.item())
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 绘制loss曲线
plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')

predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())
# 分类准确率
score = accuracy_score(y_test, predictions)

print(score)

# 保存模型
torch.save(model, 'prediction1.pt')
