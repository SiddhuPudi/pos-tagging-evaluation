from dataset.load_dataset import load_pos_dataset
from taggers.nltk_tagger import nltk_pos_tag
from taggers.spacy_tagger import spacy_pos_tag
from taggers.stanza_tagger import stanza_pos_tag
from taggers.baseline_tagger import baseline_pos_tag
from evaluation.metrics import compute_metrics
from evaluation.confusion_matrix import generate_confusion_matrix
from analysis.visualization import plot_accuracy
from analysis.results_analysis import print_summary

def calculate_accuracy(true, pred):
    correct = 0
    total = min(len(true), len(pred))
    for i in range(total):
        if true[i] == pred[i]:
            correct += 1
    return correct, total

def evaluate_model(data, tagger_function, model_name):
    total_correct = 0
    total_words = 0
    all_true = []
    all_pred = []
    for i, sample in enumerate(data):
        if i % 500 == 0:
            print(f"{model_name}: Processing sentence {i}...")
        words = sample["words"]
        true_labels = sample["labels"]
        predicted_labels = tagger_function(words)
        length = min(len(true_labels), len(predicted_labels))
        for j in range(length):
            if true_labels[j] == predicted_labels[j]:
                total_correct += 1
        total_words += length
        all_true.extend(true_labels[:length])
        all_pred.extend(predicted_labels[:length])
    accuracy = total_correct/total_words
    precision, recall, f1 = compute_metrics(all_true, all_pred)
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    return accuracy, precision, recall, f1, all_true, all_pred

def main():
    print("Starting POS Tagging Evaluation Project...\n")
    data = load_pos_dataset()
    baseline_metrics = evaluate_model(data, baseline_pos_tag, "Baseline")    
    nltk_metrics = evaluate_model(data, nltk_pos_tag, "NLTK")
    spacy_metrics = evaluate_model(data, spacy_pos_tag, "spaCy")
    stanza_metrics = evaluate_model(data, stanza_pos_tag, "Stanza")
    _, _, _, _, true_tags, pred_tags = spacy_metrics
    generate_confusion_matrix(true_tags, pred_tags, "spaCy")
    models = ["Baseline", "NLTK", "spaCy", "Stanza"]
    accuracies = [baseline_metrics[0], nltk_metrics[0], spacy_metrics[0], stanza_metrics[0]]
    plot_accuracy(models, accuracies)
    
    print("Final Comparision:")
    print(f"Baseline Metrics: \nAccuracy: {baseline_metrics[0] :.4f}, \nPrecision: {baseline_metrics[1]:.4f}, \nRecall: {baseline_metrics[2]:.4f}, \nF1 Score: {baseline_metrics[3]:.4f}\n")
    print(f"NLTK Metrics: \nAccuracy: {nltk_metrics[0]:.4f}, \nPrecision: {nltk_metrics[1]:.4f}, \nRecall: {nltk_metrics[2]:.4f}, \nF1 Score: {nltk_metrics[3]:.4f}\n")
    print(f"spaCy Metrics: \nAccuracy: {spacy_metrics[0]:.4f}, \nPrecision: {spacy_metrics[1]:.4f}, \nRecall: {spacy_metrics[2]:.4f}, \nF1 Score: {spacy_metrics[3]:.4f}\n")
    print(f"Stanza Metrics: \nAccuracy: {stanza_metrics[0]:.4f}, \nPrecision: {stanza_metrics[1]:.4f}, \nRecall: {stanza_metrics[2]:.4f}, \nF1 Score: {stanza_metrics[3]:.4f}\n")
    print_summary()

if __name__ == "__main__":
    main()