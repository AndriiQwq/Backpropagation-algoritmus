from matplotlib import pyplot as plt

def vizualize_training_process(losses):
    """
    Visualize the training process, including loss curves and accuracy.
    """

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Progress')
    plt.show()