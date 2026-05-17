import torch as th 
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
from torch.utils.data import Dataset , DataLoader

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from utils.plotting import save_loss_plot, plot_decision_boundary

class DECISIONDATASET (Dataset):
    def __init__ (self):
        self.x = th.tensor([[1.0],[2.0],[3.0],[4.0],[6.0],[7.0],[8.0],[9.0]])
        self.y = th.tensor([[0.0],[0.0],[0.0],[0.0],[1.0],[1.0],[1.0],[1.0]])
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx] , self.y[idx]
    
dataset = DECISIONDATASET()
dataloader = DataLoader(dataset , batch_size = 4 , shuffle=True)

class Decisionmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )
    def forward(self , tensor_input):
        return self.net(tensor_input)
    
model = Decisionmodel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters() , lr = 0.01 )

losses = []

for epoch in range(1001):
    epoch_loss = 0.0
    
    for batch_x , batch_y in dataloader:
        y_pred = model(batch_x)
        loss = criterion(y_pred , batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss/len(dataloader)
    losses.append(avg_epoch_loss)
    
    if epoch%100 == 0:
        print(f"Epoch:{epoch:4d} , Loss:{avg_epoch_loss:6f}")
    
save_loss_plot(losses)
plot_decision_boundary(model , dataset.x , dataset.y)

test_val = th.tensor([[5.0]] , dtype = th.float32)
with th.no_grad():
    prob = model(test_val).item()

result = "PASS" if prob >= 0.5 else "FAIL"
print("now the model will decide if the value 5.0 is pass or fail: ")
print(result)
    