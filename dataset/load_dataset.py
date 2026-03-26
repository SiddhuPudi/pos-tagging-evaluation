from datasets import load_dataset

def load_pos_dataset(split="train"):
    dataset = load_dataset("batterydata/pos_tagging")
    data = dataset[split]
    print("\nDataset Loaded Successfully!\n")
    print(f"Total sentences: {len(data)}")
    sample = data[0]
    print("\nSample Data:\n")
    print("Words:", sample["words"])
    print("Tags:", sample["labels"])
    return dataset

if __name__ == "__main__":
    load_pos_dataset()