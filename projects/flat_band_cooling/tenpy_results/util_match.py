from pathlib import Path
import numpy as np
import pandas as pd
import os

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

    geometry = parts[0]
    spin = parts[1]
    interacting = parts[2]

    params = {
        "geometry": geometry,
        "spin": spin,
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

def invert_n_to_mu(mu_vals, beta_vals, n_grid, n_target, fill_value = np.nan):
    """
    For each beta, find mu such that n(mu, beta) = n_target using 1D interpolation
    along the mu-axis.

    Parameters
    ----------
    mu_vals : 1D array, shape (N_mu,)
    beta_vals : 1D array, shape (N_beta,)
    n_grid : 2D array, shape (N_mu, N_beta)
        n_grid[i, j] = n(mu_vals[i], beta_vals[j])
    n_target : float
        Desired filling n_s or n_r.
    fill_value : float
        Value to put where n_target is outside the n(mu, beta) range.

    Returns
    -------
    mu_of_beta : 1D array, shape (N_beta,)
        mu(beta) such that n(mu(beta), beta) ≈ n_target. Entries may be NaN.
    """
    mu_vals = np.asarray(mu_vals)
    beta_vals = np.asarray(beta_vals)
    n_grid = np.asarray(n_grid)

    if n_grid.shape != (mu_vals.size, beta_vals.size):
        raise ValueError("n_grid must have shape (len(mu_vals), len(beta_vals)).")

    mu_of_beta = np.full(beta_vals.shape, fill_value, dtype=float)

    for j in range(beta_vals.size):
        n_vs_mu = n_grid[:, j]

        # Check if target is within the range for this beta
        n_min, n_max = np.nanmin(n_vs_mu), np.nanmax(n_vs_mu)
        if not (n_min <= n_target <= n_max):
            # cannot invert safely at this beta - can add extrapolation in the future
            continue

        # Ensure monotonicity for np.interp: if decreasing, flip both arrays
        # (We know that n has to increase with larger mu though, so most likely this doesn't do anything)
        if n_vs_mu[0] > n_vs_mu[-1]:
            n_sorted = n_vs_mu[::-1]
            mu_sorted = mu_vals[::-1]
        else:
            n_sorted = n_vs_mu
            mu_sorted = mu_vals

        mu_of_beta[j] = np.interp(n_target, n_sorted, mu_sorted)

    return mu_of_beta

def s2_at_mu(mu_vals, beta_vals, s2_grid, mu_of_beta, fill_value=np.nan):
    """
    Given mu(beta), get s2(mu(beta), beta) by 1D interpolation in mu for each beta.

    Parameters
    ----------
    mu_vals : 1D array, shape (N_mu,)
    beta_vals : 1D array, shape (N_beta,)
    s2_grid : 2D array, shape (N_mu, N_beta)
    mu_of_beta : 1D array, shape (N_beta,)

    Returns
    -------
    s2_of_beta : 1D array, shape (N_beta,)
    """
    mu_vals = np.asarray(mu_vals)
    beta_vals = np.asarray(beta_vals)
    s2_grid = np.asarray(s2_grid)
    mu_of_beta = np.asarray(mu_of_beta)

    if s2_grid.shape != (mu_vals.size, beta_vals.size):
        raise ValueError("s2_grid must have shape (len(mu_vals), len(beta_vals)).")

    s2_of_beta = np.full(beta_vals.shape, fill_value, dtype=float)

    for j, mu in enumerate(mu_of_beta):
        if np.isnan(mu):
            continue

        s2_vs_mu = s2_grid[:, j]
        # interpolate in mu; do NOT extrapolate beyond the grid
        s2_of_beta[j] = np.interp(mu, mu_vals, s2_vs_mu,
                                  left=fill_value, right=fill_value)

    return s2_of_beta

def _match_rows(df, row, cols, float_tol=1e-8):
    """Return boolean mask of rows in df that match `row` on all `cols`."""
    if df.empty or len(cols) == 0:
        return np.zeros(len(df), dtype=bool)

    mask = np.ones(len(df), dtype=bool)

    for col in cols:
        val = row[col]
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            # numeric comparison with tolerance, NaN == NaN
            mask &= np.isclose(
                s.to_numpy(dtype=float),
                float(val),
                atol=float_tol,
                rtol=0.0,
                equal_nan=True,
            )
        else:
            # non-numeric: exact match, treating NaN == NaN
            if pd.isna(val):
                mask &= s.isna().to_numpy()
            else:
                mask &= (s == val).to_numpy()

    return mask

def update_eos_csv(
    df_new: pd.DataFrame,
    csv_path: Path,
    param_cols: list,
    bool_overwrite_conflict: bool = False,
    float_tol: float = 1e-8,
):
    """
    Merge df_new into a CSV at csv_path. *This is currently buggy*

    param_cols:
        Columns that define the *input parameters* (including file_mtime).

    Logic per new row:
        - Look for rows in existing CSV with exactly the same param_cols.
        - If none exist: append the new row.
        - If one or more exist:
            * Compare "result columns" (all columns not in param_cols).
            * If any existing row has identical results (within float_tol): do nothing.
            * Otherwise:
                - if bool_overwrite_conflict: replace all matching rows with the new row.
                - else: print a warning, keep existing rows, ignore new row.
    """
    # 1. Load existing CSV, or start from empty
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
    else:
        df_old = pd.DataFrame()

    # 2. Align columns between old and new
    all_cols = sorted(set(df_old.columns) | set(df_new.columns))
    df_old = df_old.reindex(columns=all_cols)
    df_new = df_new.reindex(columns=all_cols)

    # "Results" = everything that isn't an input
    result_cols = [c for c in all_cols if c not in param_cols]

    # 3. If no old data, just save and return
    if df_old.empty:
        df_new.to_csv(csv_path, index=False)
        return

    identical_counter = 0
    # 4. Process each new row
    for _, row in df_new.iterrows():
        # rows with the same input parameters (incl. mtime, since it's in param_cols)
        mask_same_inputs = _match_rows(df_old, row, param_cols, float_tol=float_tol)

        if not mask_same_inputs.any():
            df_old = pd.concat([df_old, row.to_frame().T], ignore_index=True)
            continue

        # There are existing rows with identical inputs
        existing_df = df_old.loc[mask_same_inputs]

        # Check if any of them has identical results as well
        mask_identical_results = _match_rows(
            existing_df, row, result_cols, float_tol=float_tol
        )
        if mask_identical_results.any():
            identical_counter += 1
            continue

        # If we get here: same inputs, but *no* row with same results => conflict
        print(
            "WARNING: conflicting results for identical input parameters; "
            f"Ntot={row.get('Ntot')}, beta={row.get('beta')}, "
            f"n_s={row.get('n_s')}, mtime={row.get('file_mtime')}"
        )

        if bool_overwrite_conflict:
            df_old = df_old.drop(existing_df.index)
            df_old = pd.concat([df_old, row.to_frame().T], ignore_index=True)
        else:
            pass

    if identical_counter > 0:
        print(f"{identical_counter} rows already existed in .csv")
    df_old.to_csv(csv_path, index=False)