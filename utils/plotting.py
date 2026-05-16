import matplotlib.pyplot as plt
import torch as th

def save_loss_plot(losses, filename="loss.png"):
    plt.figure() 
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(filename)
    plt.show() 
    print(f"Plot saved as {filename}")
    

def plot_decision_boundary(model, x, y, filename="decision_boundary.png"):
    plt.figure(figsize=(8, 5))
    
    # 1. Plot the actual data points
    plt.scatter(x.numpy(), y.numpy(), color='#a8c7ff', label='Actual Data', zorder=5)
    
    # 2. Generate smooth points to see the "S-Curve"
    x_range = th.linspace(0, 10, 100).reshape(-1, 1)
    with th.no_grad():
        y_range = model(x_range)
    
    # 3. Plot the model's prediction line
    plt.plot(x_range.numpy(), y_range.numpy(), color='#50fa7b', label='Model Prediction (Sigmoid)', linewidth=3)
    
    # 4. Add the "Decision Fence" at 0.5
    plt.axhline(y=0.5, color='white', linestyle='--', alpha=0.3, label='Decision Boundary (0.5)')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("DECISION PLOT")
    plt.legend()
    plt.grid(alpha=0.1)
    plt.savefig(filename)
    plt.close()
    print(f"Decision plot saved as {filename}")