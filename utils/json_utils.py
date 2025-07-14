import json
import ast
import re
from typing import Any

def safe_json_loads(data: str) -> Any:
    """Safely parse JSON from a possibly malformed string.

    Tries ``json.loads`` first. If that fails, attempts ``ast.literal_eval`` and
    finally extracts the first JSON object found in the string.
    Returns an empty dict on failure.
    """
    if data is None:
        return {}
    try:
        return json.loads(data)
    except Exception:
        pass
    try:
        return ast.literal_eval(data)
    except Exception:
        pass
    match = re.search(r"{.*}", data, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}
