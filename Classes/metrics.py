from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np

def metrics(y_true, y_pred):
    """
    Info:
        Calcula as métricas de acurácia, precisão, recall, f1 e kappa.
    Params:
        y_true (array numpy): Valores corretamente marcados.
        y_pred (array numpy): Valores preditos.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return accuracy, precision, recall, f1, kappa
