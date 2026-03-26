from dataset.load_dataset import load_pos_dataset
from taggers.nltk_tagger import nltk_pos_tag

def calculate_accuracy(true, pred):
    correct = 0
    total = len(true)
    for i in range(total):
        if true[i] == pred[i]:
            correct += 1
    return correct, total

def main():
    print("Starting POS Tagging Evaluation Project...\n")
    data = load_pos_dataset()
    total_correct = 0
    total_words = 0

    for i, sample in enumerate(data):
        if i % 500 == 0:
            print(f"Processing sentence {i}...")
        words = sample["words"]
        true_labels = sample["labels"]
        predicted_labels = nltk_pos_tag(words)
        correct, total = calculate_accuracy(true_labels, predicted_labels)
        total_correct += correct
        total_words += total
    final_accuracy = total_correct / total_words
    print("\nFinal NLTK Accuracy on Dataset:")
    print(f"{final_accuracy:.4f}")

    with open("report/results.txt", "a") as f:
        f.write(f"NLTK Accuracy: {final_accuracy:.4f}\n")

if __name__ == "__main__":
    main()