from datasets import load_dataset

def load_pos_dataset(split="train"):
    dataset = load_dataset("batterydata/pos_tagging")
    data = dataset[split]
    print("\nDataset Loaded Successfully!\n")
    print(f"Total sentences: {len(data)}")
    return data

if __name__ == "__main__":
    load_pos_dataset()