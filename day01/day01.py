import torch
import matplotlib.pyplot as plt 

x = torch.tensor([2.0,4.0])
y = torch.tensor([4.0,6.0])


losses = [] 

w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)

for epoch in range(200):
    y_pred = x * w+b
    
    loss = ((y_pred - y) **2).mean()
    loss.backward()
    
    with torch.no_grad():
        w -= 0.05*w.grad
        b -= 0.05*b.grad
    w.grad.zero_()
    b.grad.zero_() 
    
    print(f"Epoch {epoch}, Loss: {loss.item()}, w: {w.item()}, b: {b.item()}")
    losses.append(loss.item())
    
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.savefig("loss.png")

plt.show()