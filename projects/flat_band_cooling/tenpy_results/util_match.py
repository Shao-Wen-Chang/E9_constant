from pathlib import Path
import numpy as np

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