from dataset.load_dataset import load_pos_dataset
from taggers.nltk_tagger import nltk_pos_tag
from taggers.spacy_tagger import spacy_pos_tag
from taggers.stanza_tagger import stanza_pos_tag
from taggers.baseline_tagger import baseline_pos_tag

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
    for i, sample in enumerate(data):
        if i % 500 == 0:
            print(f"{model_name}: Processing sentence {i}...")
        words = sample["words"]
        true_labels = sample["labels"]
        predicted_labels = tagger_function(words)
        correct, total = calculate_accuracy(true_labels, predicted_labels)
        total_correct += correct
        total_words += total
    accuracy = total_correct/total_words
    print(f"\n{model_name} Final accuracy: {accuracy:.4f}\n")
    return accuracy

def main():
    print("Starting POS Tagging Evaluation Project...\n")
    data = load_pos_dataset()
    baseline_acc = evaluate_model(data, baseline_pos_tag, "Baseline")    
    nltk_acc = evaluate_model(data, nltk_pos_tag, "NLTK")
    spacy_acc = evaluate_model(data, spacy_pos_tag, "spaCy")
    stanza_acc = evaluate_model(data, stanza_pos_tag, "Stanza")
    print("Final Comparision:")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"NLTK Accuracy: {nltk_acc:.4f}")
    print(f"spaCy Accuracy: {spacy_acc:.4f}")
    print(f"Stanza Accuracy: {stanza_acc:.4f}")

if __name__ == "__main__":
    main()