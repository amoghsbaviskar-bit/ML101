import torch as th
import torch.nn as nn
import torch.optim as optim
from utils.plotting import save_loss_plot 

x = th.tensor([[2.0],[4.0]])
y = th.tensor([[1.0],[3.0]])

class neuronNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,10)
        
        self.relu = nn.ReLU()
        
        self.layer2 = nn.Linear(10,1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        
        return x
    
model = neuronNet()


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

losses = []

for epoch in range (1000):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch:{epoch} , Loss:{loss.item()}")
    
save_loss_plot(losses)