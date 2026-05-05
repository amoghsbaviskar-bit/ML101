import torch as tr
import matplotlib.pyplot as plt

x = tr.tensor([2.0,4.0])
y = tr.tensor([1.0,3.0])
losses = []
# print(x.shape)
# print(y.shape)
w1 = tr.tensor(0.0, requires_grad=True)
w2 = tr.tensor(0.0, requires_grad=True)
b = tr.tensor(0.0, requires_grad=True)

for epoch in range(200):
    y_pred = w1*x**2+w2*x+b
    
    loss = ((y_pred - y )**2).mean()
    loss.backward()
    
    with tr.no_grad():
        w1 -= 0.003*w1.grad
        w2 -= 0.003*w2.grad
        b -= 0.003*b.grad
    w1.grad.zero_()
    w2.grad.zero_()
    b.grad.zero_()
    print(f"Epoch {epoch}, Loss: {loss.item()}, w1: {w1.item()},w2: {w2.item()}, b: {b.item()}")
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.savefig("loss.png")

plt.show()

 