from typing import Any, Dict, List

from cogkit import GenerationMode


def get_resolutions(task: GenerationMode) -> List[str]:
    if task == GenerationMode.TextToImage:
        return [
            "512x512",
            "512x768",
            "512x1024",
            "720x1280",
            "768x768",
            "1024x1024",
            "1080x1920",
        ]
    elif task == GenerationMode.TextToVideo:
        return [
            "49x480x720",
            "81x768x1360",
        ]


def flatten_dict(d: Dict[str, Any], ignore_none: bool = False) -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single layer dictionary.

    Args:
        d: The dictionary to flatten
        ignore_none: If True, keys with None values will be omitted

    Returns:
        A flattened dictionary

    Raises:
        ValueError: If there are duplicate keys across nested dictionaries
    """
    result = {}

    def _flatten(current_dict, result_dict):
        for key, value in current_dict.items():
            if value is None and ignore_none:
                continue

            if isinstance(value, dict):
                _flatten(value, result_dict)
            else:
                if key in result_dict:
                    raise ValueError(f"Duplicate key '{key}' found in nested dictionary")
                result_dict[key] = value

    _flatten(d, result)
    return result
