from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from datasets import Dataset, load_dataset

from cogkit import GenerationMode

_QUICKSTART_ROOT_DIR = Path(__file__).parent.parent.parent / "quickstart"
_GRADIO_ROOT_DIR = Path(__file__).parent.parent

_DATASET_ROOT_DIR = _QUICKSTART_ROOT_DIR / "data"
_TRAINING_SCRIPT_FILE = _QUICKSTART_ROOT_DIR / "scripts" / "train.py"
_LORA_CHECKPOINT_ROOT_DIR = _GRADIO_ROOT_DIR / "lora_checkpoints"
_CONFIG_ROOT_DIR = _GRADIO_ROOT_DIR / "configs"


def resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def get_dirs(dir_path: Path) -> List[str]:
    dir_path.mkdir(exist_ok=True)
    return [resolve_path(d) for d in dir_path.iterdir() if d.is_dir()]


def get_dataset_dirs() -> List[str]:
    """Get all dataset directories from quickstart/data."""
    return get_dirs(_DATASET_ROOT_DIR)


def get_training_script() -> str:
    return resolve_path(_TRAINING_SCRIPT_FILE)


def get_lora_checkpoint_dirs(task: GenerationMode) -> List[str]:
    """Get all lora checkpoint directories from lora_checkpoints."""
    return get_dirs(_LORA_CHECKPOINT_ROOT_DIR / task.value)


def get_lora_checkpoint_rootdir(task: GenerationMode) -> str:
    return resolve_path(_LORA_CHECKPOINT_ROOT_DIR / task.value)


def load_config_template(generation_task: GenerationMode) -> Dict[str, Any]:
    """
    Read YAML configuration template based on generation task.

    Args:
        generation_task: Task type (e.g., 't2i', 't2v')

    Returns:
        Parsed YAML as dictionary
    """
    config_file = f"{generation_task.value}.yaml"
    config_path = _CONFIG_ROOT_DIR / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    return yaml_dict


def load_data(
    data_dir: str, task: GenerationMode, split: Literal["train", "test"] = "train"
) -> Dataset:
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    assert split in ["train", "test"], f"Invalid split: {split}"
    if split == "train":
        assert train_dir.exists(), f"Train directory {train_dir} does not exist"
    else:
        assert test_dir.exists(), f"Test directory {test_dir} does not exist"

    match task:
        case GenerationMode.TextToImage:
            if split == "train":
                return load_dataset("imagefolder", data_dir=train_dir, split="train")
            else:
                return load_dataset("json", data_dir=test_dir, split="test")

        case GenerationMode.TextToVideo:
            if split == "train":
                return load_dataset("videofolder", data_dir=train_dir, split="train")
            else:
                return load_dataset("json", data_dir=test_dir, split="test")

        case GenerationMode.ImageToVideo:
            raise NotImplementedError("Image to video is not implemented")

        case _:
            raise ValueError(f"Unsupported task: {task}")
