import numpy as np

def to_current_density(values, unit, area_cm2=None):
    """Return current density in mA/cm²."""
    x = np.asarray(values, dtype=float)

    if unit == "A/cm²":
        return x * 1000.0

    if unit == "mA/cm²":
        return x

    if unit in ("A", "mA"):
        if area_cm2 is None or area_cm2 <= 0:
            raise ValueError("A positive electrode area is required.")

        if unit == "A":
            return x * 1000.0 / area_cm2

        return x / area_cm2

    raise ValueError(f"Unsupported current unit: {unit}")
