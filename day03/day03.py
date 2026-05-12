import torch as th
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as mlt

x = th.tensor([2.0,4.0])
y = th.tensor([3.0,5.0])

class quadmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(th.tensor(0.0))
        self.w2 = nn.Parameter(th.tensor(0.0))
        self.b = nn.Parameter(th.tensor(0.0))
        
        
    def forward (self, x):
        return self.w1*x**2 + self.w2*x + self.b
    
model = quadmodel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1)

losses = []

for epoch in range (500):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch:{epoch},Loss:{loss.item()}")
mlt.plot(losses)
mlt.xlabel("Epoch")
mlt.ylabel("Loss")
mlt.title("Training Loss Curve")

mlt.savefig("loss.png")

mlt.show()
