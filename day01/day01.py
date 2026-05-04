import torch

x = torch.tensor([2.0,4.0])
y = torch.tensor([4.0,6.0])

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