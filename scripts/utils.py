from sklearn.metrics import roc_auc_score, log_loss
import numpy as np

def find_ngrams(input_list, n):
    """
    Returns the n-grams for a list of strings
    """
    return zip(*[input_list[i:] for i in range(n)])

def count_similar(q1, q2, n):
    """
    Counts the number of similar n-grams between two lists of strings
    
    Arguments
    ---------
    q1: list
        List of Strings
    q2: list
        List of Strings
    n: int
        Which n-gram to search
        
    Returns
    -------
    out: int
        Number of n-grams in common between the two lists
    """
    n1 = find_ngrams(q1, n)
    n2 = find_ngrams(q2, n)
    return len(list(set(n1).intersection(n2)))

def score_model(y_true, y_pred, y_pred_proba):
    if (y_true.shape[0] != y_pred.shape[0]) or (y_true.shape[0] != y_pred_proba.shape[0]):
        raise ValueError("Predictions and labels must have the same dimensions")
    accuracy_score = np.sum(y_true == y_pred) / y_true.shape[0]
    auc_score = roc_auc_score(y_true, y_pred_proba)
    log_score = log_loss(y_true, y_pred_proba)
    return accuracy_score, auc_score, log_score

def generate_report(y_true, y_pred, y_pred_proba):
    accuracy_score, auc_score, log_loss_score = score_model(y_true, y_pred, y_pred_proba)
    print('Loss Report')
    print('-' * 11)
    print('')
    print('Accuracy Score: {:.4f}'.format(accuracy_score))
    print('ROC  AUC Score: {:.4f}'.format(auc_score))
    print('Log Loss Score: {:.4f}'.format(log_loss_score))
    return accuracy_score, auc_score, log_loss_score