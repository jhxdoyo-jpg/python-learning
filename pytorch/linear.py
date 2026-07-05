import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
torch.manual_seed(66)
true_k=5
true_b=10
x=torch.rand(100,1)*10
y=true_k*x+true_b+torch.randn(100,1)*2
model=torch.nn.Linear(in_features=1,out_features=1)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
for w in range(200):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


pred=model(x).detach()
plt.scatter(x.numpy(),y.numpy(),label="真实数据")
plt.plot(x.numpy(),pred.numpy(),'r',label='拟合直线')
plt.legend()
plt.show()
print(f"学到的权重: {model.weight.item():.4f}")
print(f"学到的偏置: {model.bias.item():.4f}")

