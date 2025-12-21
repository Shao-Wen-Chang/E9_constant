import numpy as np
import pandas as pd
from pathlib import Path

import numpy as np
import pandas as pd

def invert_NS_to_beta_param_df(
    df,
    N_target,
    S_targets,
    param_str_list: str | list[str],
    *,
    beta_min=None,
    beta_max=None,
    fill_value=np.nan,
):
    """
    Invert (Ntot, beta) -> (param, S_tot) to get
    (Ntot, S_tot) -> (beta, param), using df (e.g. df_matching_rslt).

    We invert along beta for a fixed N_target, optionally restricted
    to beta_min <= beta <= beta_max (to select a monotonic branch).

    Parameters
    ----------
    df : DataFrame
        Must contain columns: "Ntot", "beta", "S2_tot", param_str.
    N_target : float
        Desired total particle number (must match some Ntot in df).
    S_targets : float or np.array
        Desired S_tot.
    param_str_list : str or list of str
        Column name of the parameter of interest (e.g. "V_offset", "mu_glob", ...).
    beta_min, beta_max : float or None
        Optional bounds to restrict the beta range used for the inversion.
        For example, use beta_min=0 to ignore negative temperatures.
    fill_value : float
        Returned when inversion is not possible.

    Returns
    -------
    beta_star, param_star : float, float
        beta(N_target, S_target), param(N_target, S_target),
        or (fill_value, fill_value) if not found.
    """
    if type(param_str_list) == str:
        param_str_list = [param_str_list]
    fill_value_arr = np.full_like(S_targets, fill_value)

    # 1) select rows for this N_target
    N_vals = df["Ntot"].to_numpy(dtype=float)
    mask_N = np.isclose(N_vals, N_target)

    if beta_min is not None:
        mask_N &= df["beta"].to_numpy(dtype=float) >= beta_min
    if beta_max is not None:
        mask_N &= df["beta"].to_numpy(dtype=float) <= beta_max

    df_N = df.loc[mask_N, ["beta", "S2_tot"] + param_str_list].dropna()
    if df_N.shape[0] < 2:
        # not enough points to interpolate
        return fill_value_arr, fill_value_arr

    # 2) sort by beta (scan parameter)
    df_N = df_N.sort_values("beta")

    beta  = df_N["beta"].to_numpy(dtype=float)
    S     = df_N["S2_tot"].to_numpy(dtype=float)
    param_list = []
    for param_str in param_str_list:
        param_list.append(df_N[param_str].to_numpy(dtype=float))

    # remove any NaNs
    mask_finite = np.isfinite(beta) & np.isfinite(S)
    for param in param_list:
        mask_finite &= np.isfinite(param)
    
    beta, S = beta[mask_finite], S[mask_finite]
    for param in param_list:
        param = param[mask_finite]

    if beta.size < 2:
        return fill_value_arr, fill_value_arr

    # 3) check (optional) monotonicity of S(beta) on this branch
    dS = np.diff(S)
    if not (np.all(dS >= 0) or np.all(dS <= 0)):
        # not strictly monotonic -> the "inverse" is multi-valued;
        # you can decide to warn or handle more carefully here.
        # For now we still do a simple interp, but be aware it's ambiguous.
        # print("Warning: S(beta) not monotonic on this branch; inversion ambiguous.")
        pass

    # 4) invert S(beta) -> beta(S)
    # np.interp requires S to be increasing; flip if needed
    if S[0] > S[-1]:
        S_sorted = S[::-1]
        beta_sorted = beta[::-1]
    else:
        S_sorted = S
        beta_sorted = beta

    beta_star = np.interp(S_targets, S_sorted, beta_sorted,
                            left=fill_value, right=fill_value)

    # 5) now get param at that beta
    param_star_list = []
    for param in param_list:
        param_star_list.append(np.interp(beta_star, beta, param,
                            left=fill_value, right=fill_value))

    return beta_star, param_star_list

def invert_NS_mat_df(
    df,
    N_targets,
    S_targets,
    param_str_list,
    *,
    beta_min=None,
    beta_max=None,
    fill_value=np.nan,
):
    """
    Vectorized wrapper: invert a list/array of N_targets and S_targets.

    Returns
    -------
    betas : 2D array
    params  : 2D array
        beta(S), Delta mu(S) for each S_targets[i]
    """
    if type(param_str_list) == str:
        param_str_list = [param_str_list]
    
    N_targets   = np.atleast_1d(N_targets)
    S_targets   = np.atleast_1d(S_targets)
    betas       = np.full([len(N_targets), len(S_targets)], fill_value, dtype=float)
    params_list = [np.full([len(N_targets), len(S_targets)], fill_value, dtype=float) for _ in param_str_list]
    
    for i, N_target in enumerate(N_targets):
        betas[i,:], params_at_N_tar = invert_NS_to_beta_param_df(
            df,
            N_target = N_target,
            S_targets = S_targets,
            param_str_list = param_str_list,
            beta_min = beta_min,
            beta_max = beta_max,
            fill_value = fill_value,
        )
        for j, params in enumerate(params_list):
            params[i,:] = params_at_N_tar[j]

    return betas, params_list

def parametric_slice_2D(
    axis0,
    axis1,
    arr_data,
    arr_param,
    V0,
    fill_value=np.nan,
    atol=1e-10,
):
    """
    Given arr_data(x, y) and param(x, y) on a rectangular grid, extract param along the
    constant-V curve V(x, y) = V0, assuming N(S)|(V = V0) is single-valued.

    Parameters
    ----------
    axis0 : 1D array, shape (N_N,)
        Grid values for N (axis 0 of arr_data, arr_param).
    axis1 : 1D array, shape (N_S,)
        Grid values for S (axis 1 of arr_data, arr_param).
    arr_data : 2D array, shape (N_N, N_S)
        arr_data[i, j] = V(axis0[i], axis1[j]).
    arr_param : 2D array, shape (N_N, N_S)
        arr_param[i, j] = s_s(axis0[i], axis1[j]).
    V0 : float
        Target value of V defining the curve V = V0.
    fill_value : float (default: NaN)
        Value used when no crossing is found at a given S.
    atol : float
        Tolerance for considering V == V0 at a grid point.

    Returns
    -------
    N_on_curve : 1D array, shape (N_S,)
        N(S) such that V(N(S), S) = V0. May contain fill_value where no solution.
    s_s_on_curve : 1D array, shape (N_S,)
        s_s(N(S), S) along the constant-V curve. Same masking as N_on_curve.
    """
    axis0 = np.asarray(axis0, dtype=float)
    axis1 = np.asarray(axis1, dtype=float)
    arr_data = np.asarray(arr_data, dtype=float)
    arr_param = np.asarray(arr_param, dtype=float)

    if arr_data.shape != arr_param.shape:
        raise ValueError("arr_data and arr_param must have the same shape.")
    if arr_data.shape != (axis0.size, axis1.size):
        raise ValueError(
            "arr_data shape must be (len(axis0), len(axis1)); "
            f"got {arr_data.shape} vs ({axis0.size}, {axis1.size})."
        )

    N_on_curve = np.full(axis1.shape, fill_value, dtype=float)
    s_s_on_curve = np.full(axis1.shape, fill_value, dtype=float)

    # Loop over S; for each S_j, treat V(N, S_j) as a 1D function of N
    for j in range(axis1.size):
        V_col = arr_data[:, j]
        s_col = arr_param[:, j]

        # Skip columns with no finite data
        if not np.any(np.isfinite(V_col)):
            continue

        diff = V_col - V0

        # 1) Check for exact (within atol) hit on a grid point
        exact_idx = np.where(np.isfinite(diff) & (np.abs(diff) <= atol))[0]
        if exact_idx.size > 0:
            i0 = exact_idx[0]
            N0 = axis0[i0]
            s0 = s_col[i0]
            N_on_curve[j] = N0
            s_s_on_curve[j] = s0
            continue

        # 2) Look for a sign change of V - V0 between neighboring N points
        found = False
        for i in range(axis0.size - 1):
            if not (np.isfinite(diff[i]) and np.isfinite(diff[i+1])):
                continue

            # Check if V - V0 changes sign between i and i+1
            if diff[i] * diff[i+1] < 0:
                # Linear interpolation in N between (N_i, V_i) and (N_{i+1}, V_{i+1})
                N0 = np.interp(
                    V0,
                    [V_col[i], V_col[i+1]],
                    [axis0[i], axis0[i+1]],
                )
                # Interpolate s_s at that N0 using the same bracket
                s0 = np.interp(
                    N0,
                    [axis0[i], axis0[i+1]],
                    [s_col[i], s_col[i+1]],
                )
                N_on_curve[j] = N0
                s_s_on_curve[j] = s0
                found = True
                break

        # If no sign change found, N_on_curve[j] and s_s_on_curve[j]
        # remain as fill_value (e.g. NaN)
        if not found:
            continue

    return N_on_curve, s_s_on_curve

#%% Obsolete
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