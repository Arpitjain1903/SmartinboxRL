def safe_score(value) -> float:
    """Clamp *value* into the strictly-open interval (0.01, 0.99).

    HARD REQUIREMENT: score must be strictly between 0 and 1.
    Handles None, NaN, bool, and any numeric type.
    NEVER returns exactly 0.0 or 1.0 — satisfies the hackathon hard requirement.
    """
    if value is None or value != value:  # handles None and NaN
        return 0.5
    return max(0.01, min(0.99, float(value)))
