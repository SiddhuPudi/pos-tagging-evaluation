from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def generate_confusion_matrix(true_tags, pred_tags, model_name):
    labels = list(set(true_tags))
    cm = confusion_matrix(true_tags, pred_tags, labels=labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{model_name}_confusion_matrix.png")