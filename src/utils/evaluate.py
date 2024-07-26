import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, save_path: Optional[Path] = None):
    """
    Plot the confusion matrix and save it as a PNG image.

    :param y_true: the actual labels
    :param y_pred: the model predictions
    :param save_path: the path to save the plot, defaults to None
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Visualize the confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", cbar=False)  # Remove color bar

    # Add true and predicted labels to the confusion matrix plot
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.xticks([0.5, 1.5], ["Not Adopted", "Adopted"])  # Move ticks to the middle of the squares
    plt.yticks([0.5, 1.5], ["Not Adopted", "Adopted"])  # Move ticks to the middle of the squares
    plt.show()

    # Save the plot as a PNG image
    if save_path:
        plt.savefig(save_path)
    plt.close()


def classification_metrics(y_true: pd.Series, y_pred: pd.Series, mode: str):
    """
    Calculate classification metrics for the model predictions.

    :param y_true: the actual labels
    :param y_pred: the model predictions
    :param mode: a string indicating the mode (train, val, or test)
    :return: a dictionary containing the computed metrics
    """
    metrics = {}

    metrics[f"{mode}_accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{mode}_precision"] = precision_score(y_true, y_pred)
    metrics[f"{mode}_recall"] = recall_score(y_true, y_pred)
    metrics[f"{mode}_f1_score"] = f1_score(y_true, y_pred)
    metrics[f"{mode}_roc_auc"] = roc_auc_score(y_true, y_pred)

    return metrics
