from typing import Optional

from datasets import DatasetDict, load_dataset


def create_and_save_dataset(
    train_json_path,
    test_json_path: Optional[str] = None,
    output_dir: str = "./dataset/store",
):
    """
    Load train and test JSON files, combine them into a DatasetDict, and save to disk.

    Parameters:
    train_json_path (str): Path to the training JSON file.
    test_json_path (str): Path to the testing JSON file.
    output_dir (str): Directory where the combined dataset will be saved.
    """
    # Load the JSON files
    train_dataset = load_dataset("json", data_files={"train": train_json_path})
    if test_json_path:
        test_dataset = load_dataset("json", data_files={"test": test_json_path})
        # Combine into a DatasetDict
        dataset = DatasetDict(
            {"train": train_dataset["train"], "test": test_dataset["test"]}
        )
    else:
        dataset = train_dataset

    # Save the combined dataset to disk
    dataset.save_to_disk(output_dir)
    return dataset


# Example usage
# create_and_save_dataset('dataset/yoga_beginner/train.json', 'dataset/yoga_beginner/test.json', 'dataset/yoga_beginner/yoga_beginner_dataset')
