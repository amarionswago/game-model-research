"""Small, deterministic research reports, including Transformers metadata. AI-authored."""
from collections.abc import Mapping
import json
import math
from pathlib import Path


def json_record(value):
    """Normalize metadata without silently stringifying tensors or unknown objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'nonfinite_float': repr(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('Report dictionary keys must be strings')
        return {key: json_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_record(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_record(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(f'Unsupported report value type: {type(value).__name__}')


def write_report(path, report):
    """Validate first, then replace atomically so a failure preserves the last report."""
    path = Path(path)
    text = json.dumps(json_record(report), indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)
