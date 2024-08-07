import pandas as pd
from pytest import approx

from pet_adoption.evaluate import classification_metrics, plot_classification_report


def test_plot_classification_report():
    y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 0, 1, 1, 0, 0])
    report_dict = plot_classification_report(y_true, y_pred)
    assert isinstance(report_dict, dict)
    assert report_dict["Not Adopted"]["precision"] == 0.6
    assert report_dict["Not Adopted"]["recall"] == 0.75
    assert report_dict["Adopted"]["recall"] == 0.5
    assert report_dict["accuracy"] == 0.625


def test_classification_metrics():
    mode = "train"
    y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 0, 1, 1, 0, 0])
    metrics = classification_metrics(y_true, y_pred, mode=mode)
    assert metrics[f"{mode}_accuracy"] == 0.625
    assert metrics[f"{mode}_precision"] == approx(0.67, 0.1)
    assert metrics[f"{mode}_recall"] == 0.5
    assert metrics[f"{mode}_roc_auc"] == 0.625
