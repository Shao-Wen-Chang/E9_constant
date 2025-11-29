import numpy as np
import pandas as pd
import glob
from pathlib import Path

def parse_folder_name(folder_name: str) -> dict:
    """
    Parse a folder name like
    'sawtooth_spinless_I_L25_t1.0_tp1.41421356237_shift0.0_ymax0_V1.0_mu-0.3_dBeta0.1_Nsteps200_chi64_svd1e-08'
    into a dictionary of parameters.
    """
    name = Path(folder_name).name  # in case you pass a full path
    parts = name.split('_')

    if len(parts) < 3:
        raise ValueError(f"Folder name doesn't have expected format: {folder_name}")

    lattice_geometry = parts[0]
    spin_part = parts[1]
    interact_part = parts[2]

    # Spin: True = spinful, False = spinless
    if spin_part == "spinful":
        spinful = True
    elif spin_part == "spinless":
        spinful = False
    else:
        raise ValueError(f"Unknown spin specifier: {spin_part}")

    # Interacting: True = I, False = NI
    if interact_part == "I":
        interacting = True
    elif interact_part == "NI":
        interacting = False
    else:
        raise ValueError(f"Unknown interaction specifier: {interact_part}")

    params = {
        "lattice_geometry": lattice_geometry,
        "spinful": spinful,
        "interacting": interacting,
    }

    # Parse remaining key/value chunks
    for chunk in parts[3:]:
        # find first numeric-ish character
        split_idx = None
        for i, ch in enumerate(chunk):
            if ch.isdigit() or ch in "+-.":
                split_idx = i
                break

        if split_idx is None:
            # no numeric part found; skip or store raw
            continue

        key = chunk[:split_idx]
        val_str = chunk[split_idx:]

        # try int, then float
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str  # fallback, unlikely given your format

        params[key] = val

    return params

def get_eos_file_path(params_dict, subdir = "eos_results"):
    pattern_parts = ["eos"]
    # Loop through the parameters in params_dict and construct the pattern
    for param, value in params_dict.items():
        # If value is a wildcard, use '*' in the pattern; otherwise, format the value
        if value == "*":
            pattern_parts.append(f"_{param}*")
        else:
            pattern_parts.append(f"_{param}{value:.3f}".replace(".", "p") if isinstance(value, float) else f"_{param}{value}")
    
    pattern_parts.append(".csv")
    return Path(subdir, "".join(pattern_parts))

def add_point(records, fixed_dict, other_dict):
    rec = {}
    rec.update(fixed_dict)
    rec.update(other_dict)
    records.append(rec)

def read_eos_files(dir = ""):
    # Load all slices
    csv_path = Path(dir)
    files = sorted(csv_path.glob("eos_*.csv"))
    dfs = [pd.read_csv(f) for f in files]

    # Concatenate into one big table
    if len(dfs) == 0:
        print("no matching files found")
        eos = None
    elif len(dfs) == 1:
        eos = dfs[0]
    else:
        eos = pd.concat(dfs, ignore_index=True)

    return eos