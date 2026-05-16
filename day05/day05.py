import torch as th
import torch.nn as nn 
import torch.optim as optim
import sys
from pathlib import Path


root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from utils.plotting import save_loss_plot , plot_decision_boundary

x = th.tensor([[1.0],[2.0],[3.0],[4.0],[6.0],[7.0],[8.0],[9.0]])
 
y = th.tensor([[0.0],[0.0],[0.0],[0.0],[1.0],[1.0],[1.0],[1.0],])

class decisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        ) 
    def forward (self, tensor_input):
        return self.net(tensor_input)
    
model = decisionNet()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

losses = []

for epoch in range(1001):
    y_pred = model(x)
    loss = criterion(y_pred , y )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch%100 == 0 :
        print(f"Epoch:{epoch:4d}, loss: {loss.item():6f}")

save_loss_plot(losses)
plot_decision_boundary(model, x, y)

test_val = th.tensor([[5.0]] , dtype = th.float32)
with th.no_grad():
    prob = model(test_val).item()

result = "PASS" if prob >= 0.5 else "FAIL"
print("now the model will decide if the value 5.0 is pass or fail: ")
print(result)