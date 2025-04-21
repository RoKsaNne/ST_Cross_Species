import matplotlib.pyplot as plt
import os

def plot_separate_loss_curves(class_loss, recon_loss, mmd_loss, total_loss, save_dir):
    """Helper function to plot and save separate loss curves."""
    plt.figure(figsize=(10, 8))

    # Plot classification loss
    plt.plot(range(1, len(class_loss) + 1), class_loss, label='Classification Loss', color='blue')

    # Plot domain loss
    plt.plot(range(1, len(recon_loss) + 1), recon_loss, label='Reconstruction Loss', color='orange')

   # Plot MMD loss
    plt.plot(range(1, len(mmd_loss) + 1), mmd_loss, label='MMD Loss', color='green')

    # Plot total loss
    plt.plot(range(1, len(total_loss) + 1), total_loss, label='Total Loss', color='red')

    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "Loss_Curves.png"))
    plt.close()
    