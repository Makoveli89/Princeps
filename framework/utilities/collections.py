from typing import Any, Dict

def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges source dictionary into target dictionary.

    :param target: The dictionary to merge into.
    :param source: The dictionary to merge from.
    :return: The updated target dictionary.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries in order.
    Later configurations overwrite earlier ones.

    :param configs: Variable number of dictionaries.
    :return: A new dictionary containing the merged result.
    """
    result = {}
    for config in configs:
        deep_merge(result, config.copy())
    return result
