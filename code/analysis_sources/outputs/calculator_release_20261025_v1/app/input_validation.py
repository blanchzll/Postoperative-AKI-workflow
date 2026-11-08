import math
import re

_CASE_KEY_RE = re.compile(r'^[A-Za-z0-9_.:|+\-]+$')


def validate_records(records, schema, units):
    if not isinstance(records, list) or not records or len(records) > 1000:
        raise ValueError("records must be a nonempty list of at most 1000 dicts")

    if not isinstance(schema, list) or not schema:
        raise ValueError("schema must be a nonempty list")

    if not isinstance(units, dict):
        raise ValueError("units must be a dict")

    schema_names = []
    schema_map = {}
    for entry in schema:
        if not isinstance(entry, dict):
            raise ValueError("schema entries must be dicts")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("schema entry name must be a nonempty string")
        if name in schema_map:
            raise ValueError("duplicate schema name")
        stype = entry.get("type")
        if stype not in ("number", "category"):
            raise ValueError("schema type must be 'number' or 'category'")
        unit = entry.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise ValueError("schema unit must be a string or None")
        if stype == "category":
            allowed = entry.get("allowed")
            if allowed is not None:
                if not isinstance(allowed, (list, tuple)):
                    raise ValueError("allowed must be a list or tuple")
                for a in allowed:
                    if not isinstance(a, str):
                        raise ValueError("allowed entries must be strings")
        schema_names.append(name)
        schema_map[name] = entry

    expected_keys = set(schema_names)
    expected_keys.add("case_key")

    required_units = {}
    for name in schema_names:
        entry = schema_map[name]
        unit = entry.get("unit")
        if unit:
            required_units[name] = unit

    if set(units.keys()) != set(required_units.keys()):
        raise ValueError("units must exactly match schema non-dimensionless units")
    for name, unit in required_units.items():
        if units[name] != unit:
            raise ValueError(f"units mismatch for field {name}")

    seen_case_keys = set()
    normalized = []

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"record {idx}: must be a dict")

        rec_keys = set(rec.keys())
        if rec_keys != expected_keys:
            raise ValueError(f"record {idx}: fields do not match the selected route")

        ck = rec["case_key"]
        if not isinstance(ck, str) or not ck or len(ck) > 128:
            raise ValueError(f"record {idx} field case_key: invalid anonymous key")
        if not _CASE_KEY_RE.fullmatch(ck):
            raise ValueError(f"record {idx} field case_key: invalid anonymous key")
        if ck in seen_case_keys:
            raise ValueError(f"record {idx} field case_key: duplicate key")
        seen_case_keys.add(ck)

        norm = {"case_key": ck}
        for name in schema_names:
            entry = schema_map[name]
            stype = entry["type"]
            val = rec[name]

            if val is None:
                norm[name] = None
                continue

            if stype == "number":
                if isinstance(val, bool):
                    raise ValueError(f"record {idx} field {name}: field must be number")
                if isinstance(val, str):
                    raise ValueError(f"record {idx} field {name}: field must be number")
                if not isinstance(val, (int, float)):
                    raise ValueError(f"record {idx} field {name}: field must be number")
                try:
                    fval = float(val)
                except (OverflowError, ValueError):
                    raise ValueError(f"record {idx} field {name}: number out of range") from None
                if not math.isfinite(fval):
                    raise ValueError(f"record {idx} field {name}: field must be finite")
                lower = entry.get("lower")
                upper = entry.get("upper")
                if lower is not None:
                    if not isinstance(lower, (int, float)) or isinstance(lower, bool):
                        raise ValueError("lower bound invalid")
                    if fval < float(lower):
                        raise ValueError(f"record {idx} field {name}: field below lower bound")
                if upper is not None:
                    if not isinstance(upper, (int, float)) or isinstance(upper, bool):
                        raise ValueError("upper bound invalid")
                    if fval > float(upper):
                        raise ValueError(f"record {idx} field {name}: field above upper bound")
                norm[name] = fval
            else:
                if not isinstance(val, str):
                    raise ValueError(f"record {idx} field {name}: field must be category string")
                allowed = entry.get("allowed")
                if allowed is not None:
                    if val not in allowed:
                        raise ValueError(f"record {idx} field {name}: field category not allowed")
                norm[name] = val

        normalized.append(norm)

    return normalized