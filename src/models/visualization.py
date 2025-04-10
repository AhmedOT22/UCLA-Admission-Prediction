import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_curve(model):
    """plot_loss_curve- Visualizes the loss curve of the model during training.

    Args:
        model (_type_): _description_
    """
    if hasattr(model, "loss_curve_"):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Loss', color='blue')
        plt.title('Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Model does not have a loss_curve_.")
