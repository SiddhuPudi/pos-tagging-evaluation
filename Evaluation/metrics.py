from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(true_tags, pred_tags):
    precision = precision_score(true_tags, pred_tags, average="weighted", zero_division=0)
    recall = recall_score(true_tags, pred_tags, average="weighted", zero_division=0)
    f1 = f1_score(true_tags, pred_tags, average="weighted", zero_division=0)
    return precision, recall, f1