import json
import pickle

def preprocess_verbatim_dataset(
    dataset_path: str,
    save_path: str,
):
    """
    Preprocess the verbatim dataset into a JSONL dataset.
    """

    # Load the dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Create the JSONL dataset
    output = []

    for row in dataset:
        text = " ".join(row)

        output.append({
            "text": text,
        })
    
    # Save the dataset
    with open(save_path, "w") as f:
        for entry in output:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    preprocess_verbatim_dataset(
        dataset_path=args.dataset_path,
        save_path=args.save_path,
    )