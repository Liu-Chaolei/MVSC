from collections import defaultdict
from typing import Any


def expand_list(list_of_dict: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    """
    Expand a list of dictionaries to a dictionary of lists.
    """
    result = defaultdict(list)
    for d in list_of_dict:
        for key, values in d.items():
            result[key].extend(values)
    return dict(result)
