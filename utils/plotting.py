import matplotlib.pyplot as plt

def save_loss_plot(losses, filename="loss.png"):
    plt.figure() 
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(filename)
    plt.show() 
    print(f"Plot saved as {filename}")