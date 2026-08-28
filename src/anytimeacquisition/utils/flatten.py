def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into dotted keys, e.g. {"a": {"b": 1}} -> {"a.b": 1}."""
    items: dict = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten(v, key, sep))
        else:
            items[key] = v
    return items
