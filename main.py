from dataset.load_dataset import load_pos_dataset

def main():
    print("Starting POS Tagging Evaluation Project...\n")
    data = load_pos_dataset()
    print("\nDataset ready for processing!\n")

if __name__ == "__main__":
    main()