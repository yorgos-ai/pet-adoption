import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred):
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualize the confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    # Add true and predicted labels to the confusion matrix plot
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
