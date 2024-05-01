import torch
import yaml


def remove_module_prefix(state_dict):
    """
    Remove the "module." prefix from the keys in the state_dict.

    Args:
        state_dict (dict): A state dictionary containing the weights of a model.

    Returns:
        new_state_dict (dict): A new state dictionary with the "module." prefix removed from the keys.
    """
    new_state_dict = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }
    return new_state_dict


def load_model_weights(model, filepath, strict: bool = True):
    """
    Load model weights from the specified file path.

    Args:
        model (torch.nn.Module): The model for which the weights will be loaded.
        filepath (str): Path to the file containing the model weights.

    Returns:
        model (torch.nn.Module): The model with the loaded weights.
    """
    # Load the weights from the file
    pretrained_weights = torch.load(filepath, map_location=torch.device("cpu"))

    # Extract the model's state dictionary and remove the "module." prefix if present
    pretrained_state_dict = remove_module_prefix(pretrained_weights["model_state_dict"])

    # Extract the current model's state dictionary
    current_state_dict = model.state_dict()

    if strict:
        model.load_state_dict(pretrained_state_dict)
        return model

    # Create a new state dictionary to hold the weights to be loaded into the current model
    new_state_dict = {}

    # Iterate over the current state dictionary
    for key, current_tensor in current_state_dict.items():
        # Check if the key is in the pre-trained state dictionary and the shapes are equal
        if (
            key in pretrained_state_dict
            and current_tensor.shape == pretrained_state_dict[key].shape
        ):
            # Use the pre-trained weights
            new_state_dict[key] = pretrained_state_dict[key]
        else:
            # Use the current model's weights
            new_state_dict[key] = current_tensor

    # Load the new state dictionary into the model
    model.load_state_dict(new_state_dict)

    return model


def write_to_yaml(data, file_path):
    """
    Writes the provided dictionary data to a YAML file with UTF-8 encoding.

    Args:
    - data (dict): The dictionary to be written to a YAML file.
    - file_path (str): The path to the file where data should be written.

    Returns:
    - None
    """
    with open(file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True)


def load_from_yaml(file_path):
    """
    Loads data from a YAML file into a dictionary.

    Args:
    - file_path (str): The path to the YAML file to be read.

    Returns:
    - dict: The dictionary containing the data from the YAML file.
    """
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def read_poses_from_file(file_path):
    """
    Reads the pose data from a given file and returns a dictionary.

    Args:
    - file_path (str): The path to the file containing pose data.

    Returns:
    - dict: A dictionary where keys are pose names and values are lists of labels.
    """

    poses_dict = {}

    with open(file_path, "r") as file:
        for line in file:
            # Find the last space in the line and split from there
            split_index = line.rfind(" ")
            pose_name = line[:split_index].strip()
            labels = list(map(int, line[split_index:].strip().split(",")))
            poses_dict[pose_name] = labels

    return poses_dict
