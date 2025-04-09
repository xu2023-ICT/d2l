import torch
from torch import nn

# 模拟输入维度 / 隐层 / 输出维度
num_inputs, num_hiddens, num_outputs = 784, 256, 10
batch_size = 4
epochs = 10

# 模拟数据
X = torch.randn(batch_size, num_inputs)      # [4, 784]
y = torch.tensor([0, 1, 2, 3])               # [4]

# 初始化参数
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs))

# 定义ReLU
def relu(X):
    return torch.maximum(X, torch.tensor(0.0))

def net(X):
    H = relu(X @ W1 + b1)
    logits = H @ W2 + b2
    return logits





class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
    
    def forward(self, X):
        H = relu(X @ W1 + b1)
        logits = H @ W2 + b2
        return logits

loss_fn = nn.CrossEntropyLoss()
net = MyNet()
optimizer = torch.optim.SGD(net.paramters(), lr=1e-3)

for epoch in range(epochs):    
    logits = net(X)
    loss = loss_fn(logits, y)        # shape: [4]
    print("loss:", loss)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")

# 检查梯度是否已经被计算
print("W1 的梯度存在吗？", W1.grad is not None)
