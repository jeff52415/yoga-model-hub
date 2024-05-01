from pathlib import Path

import yaml
from pydantic import BaseModel

from yogahub import ROOT

DEFAULT_CONFIG_PATH = ROOT / "yogahub/cfg/train/classify_config.yaml"


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    device: str
    num_epochs: int
    lr: float
    label_smooth: float
    train_path: str
    test_path: str
    drop_path_rate: float
    num_workers: int
    pin_memory: bool
    batch_size: int
    weight_decay: float
    warmup_period: float
    weight_path: str
    model: str
    pretrained: str


def find_config_file() -> Path:
    """Locate the configuration file."""
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    raise Exception(f"Config not found at {DEFAULT_CONFIG_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None):
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path) as conf_file:
            parsed_config = yaml.load(conf_file, Loader=yaml.FullLoader)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config=None):
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = ModelConfig(**parsed_config)

    return _config


classify_config = create_and_validate_config()
