import matplotlib.pyplot as plt
import os

def plot_separate_loss_curves(losses_dict, save_dir):
    """Helper function to plot and save separate loss curves.

    Args:
        losses_dict (dict): Dictionary containing loss names as keys and loss values (list of values) as values.
        save_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 8))

    # Plot each loss in the dictionary
    for loss_name, loss_values in losses_dict.items():
        plt.plot(range(1, len(loss_values) + 1), loss_values, label=loss_name)

    # Customize plot appearance
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "Loss_Curves.png"))
    plt.close()

# def plot_separate_loss_curves(class_loss, recon_loss, mmd_loss, total_loss, save_dir):
#     """Helper function to plot and save separate loss curves."""
#     plt.figure(figsize=(10, 8))

#     # Plot classification loss
#     plt.plot(range(1, len(class_loss) + 1), class_loss, label='Classification Loss', color='blue')

#     # Plot domain loss
#     plt.plot(range(1, len(recon_loss) + 1), recon_loss, label='Reconstruction Loss', color='orange')

#    # Plot MMD loss
#     plt.plot(range(1, len(mmd_loss) + 1), mmd_loss, label='MMD Loss', color='green')

#     # Plot total loss
#     plt.plot(range(1, len(total_loss) + 1), total_loss, label='Total Loss', color='red')

#     plt.title('Loss Curves')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(save_dir, "Loss_Curves.png"))
#     plt.close()
    


def plot_separate_loss_curves_homo(class_loss, recon_loss, total_loss, save_dir):
    """Helper function to plot and save separate loss curves."""
    plt.figure(figsize=(10, 8))

    # Plot classification loss
    plt.plot(range(1, len(class_loss) + 1), class_loss, label='Classification Loss', color='blue')

    # Plot domain loss
    plt.plot(range(1, len(recon_loss) + 1), recon_loss, label='Reconstruction Loss', color='orange')

    # Plot total loss
    plt.plot(range(1, len(total_loss) + 1), total_loss, label='Total Loss', color='red')

    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "Loss_Curves.png"))
    plt.close()

def plot_acc_curves(acc_ref, acc_tgt, save_dir, filename='acc_curves.png'):
    """
    Plots reference vs. target accuracy over epochs and saves the figure.
    """
    epochs = range(1, len(acc_ref) + 1)

    plt.figure()
    plt.plot(epochs, acc_ref, label='Ref Acc')
    plt.plot(epochs, acc_tgt, label='Tgt Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Reference & Target Accuracy per Epoch')
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path)
    plt.close()