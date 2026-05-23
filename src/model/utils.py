import numpy as np

def custom_score(y_true, y_pred, n_var):
    # when using do:
    # from sklearn.metrics import make_scorer
    # scorer = make_scorer(custom_score, greater_is_better=True)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp * 10 - fp * 5 - n_var * 200
