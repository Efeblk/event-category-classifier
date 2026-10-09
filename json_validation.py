"""Strict UTF-8 JSON parsing shared by local HTTP and offline data readers."""

import json
import math


def strict_json(raw):
    """Reject ambiguous JSON and values that cannot be sent as valid UTF-8."""
    def object_members(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON object fields must be unique.")
            result[key] = value
        return result

    def finite_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite.")
        return number

    def invalid_constant(value):
        raise ValueError("JSON numbers must be finite.")

    def check_strings(value):
        if isinstance(value, str):
            value.encode("utf-8")
        elif isinstance(value, dict):
            for key, item in value.items():
                check_strings(key)
                check_strings(item)
        elif isinstance(value, list):
            for item in value:
                check_strings(item)

    try:
        value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw,
                           object_pairs_hook=object_members, parse_float=finite_number,
                           parse_constant=invalid_constant)
        check_strings(value)
        return value
    except UnicodeError as error:
        raise ValueError("JSON must contain valid UTF-8 text, without unpaired surrogates.") from error
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep.") from error
