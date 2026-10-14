import util_match
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path

#%% Helper functions
def load_eos_data(
    scan_folders: Path,
    tnpspc: dict = None,
    is_joint: bool = None,
) -> pd.DataFrame:
    """
    Load EoS measurements from all subfolders in scan_folder into a pandas DataFrame.

    Checks parameter consistency across folder names and measurement files,
    and checks consistency of beta grids across runs.

    Parameters
    ----------
    scan_folder : Path
        Directory containing simulation subfolders.
    tnpspc : dict, optional
        Dictionary of parameters (e.g. {'geometry': 'sawtooth', 'L': 100, 'chi': 64})
        that must be consistent across all subfolders.
    is_joint : bool, optional
        Whether the data comes from a joint simulation (with muS and muR)
        or a single-subregion simulation (with mu). If None, auto-detected from file content.

    Returns
    -------
    pd.DataFrame
        DataFrame containing records of all measurement data.
    """
    if not isinstance(scan_folders, (list, tuple)):
        scan_folders = [scan_folders]
    
    subfolder_list = []
    for scan_folder in scan_folders:
        scan_folder = Path(scan_folder)
        if not scan_folder.exists():
            raise FileNotFoundError(f"Scan folder does not exist: {scan_folder}")
        sl = [f for f in scan_folder.iterdir() if f.is_dir()]
        subfolder_list += sl
        print(f"Found {len(sl)} folders in {scan_folder.name}")

    # folder_params_shared = None
    records_EoS = []
    first_betas = None

    for sf_path in subfolder_list:
        filepath = sf_path / "measurements.pkl"
        if not filepath.is_file():
            print(f"No measurements.pkl found in {sf_path.name}")
            continue

        with open(filepath, "rb") as f:
            file_content = pickle.load(f)

        try:
            folder_params = util_match.parse_folder_name(sf_path)

            if is_joint is None:    # Autodetect
                is_joint = "muS" in file_content

            # Check mu consistency between folder name and file
            if is_joint:
                for s in ["muS", "muR"]:
                    mu = file_content[s]
                    if not np.isclose(mu, folder_params[s]):
                        raise ValueError(
                            f"{s}_folder = {folder_params[s]} doesn't agree with mu_file = {mu}"
                        )
                    folder_params.pop(s)
            else:
                mu = file_content["mu"]
                if not np.isclose(mu, folder_params["mu"]):
                    raise ValueError(
                        f"mu_folder = {folder_params['mu']} doesn't agree with mu_file = {mu}"
                    )
                folder_params.pop("mu")

            # Check shared folder parameters
            V = folder_params.pop("V")
            shift = folder_params.pop("shift")
            tp = folder_params.pop("tp")

            if tnpspc is not None:
                for k, v in tnpspc.items():
                    if folder_params.get(k) != v:
                        raise ValueError(
                            f"Mismatch in tenpy parameter {k}: specified {v}, "
                            f"but {folder_params.get(k)} from folder"
                        )

            # Save data records
            file_mtime = os.path.getmtime(filepath)
            for i_b, beta in enumerate(file_content["betas"]):
                if is_joint:
                    (
                        S2,
                        S2S,
                        S2R,
                        n_avg,
                        n_avgS,
                        n_avgR,
                        energy,
                        energyS,
                        energyR,
                        eps,
                        ov,
                        max_chi,
                        trace,
                        runtime,
                    ) = file_content["data"][i_b, :]
                    records_EoS.append({
                        "muS": file_content["muS"],
                        "muR": file_content["muR"],
                        "beta": beta,
                        "S2": S2,           # per site (total joint)
                        "S2S": S2S,         # per site (subregion S)
                        "S2R": S2R,         # per site (subregion R)
                        "n_avg": n_avg,     # per site (total joint)
                        "n_avgS": n_avgS,   # per site (subregion S)
                        "n_avgR": n_avgR,   # per site (subregion R)
                        "energy": energy,   # per site (total joint)
                        "energyS": energyS, # per site (subregion S)
                        "energyR": energyR, # per site (subregion R)
                        "eps": eps,
                        "ov": ov,
                        "max_chi": max_chi,
                        "trace": trace,
                        "runtime": runtime,
                        "file_mtime": file_mtime,
                        "V": V,
                        "shift": shift,
                        "tp": tp,
                    })
                else:
                    (
                        S2,
                        n_avg,
                        energy,
                        eps,
                        ov,
                        max_chi,
                        trace,
                        runtime,
                    ) = file_content["data"][i_b, :]
                    records_EoS.append({
                        "mu": file_content["mu"],
                        "beta": beta,
                        "S2": S2,           # per site
                        "n_avg": n_avg,     # per site
                        "energy": energy,   # per site
                        "eps": eps,
                        "ov": ov,
                        "max_chi": max_chi,
                        "trace": trace,
                        "runtime": runtime,
                        "file_mtime": file_mtime,
                        "V": V,
                        "shift": shift,
                        "tp": tp,
                    })
        except ValueError as e:
            print(f"Error reading {filepath}: {e}")
            raise e

        # Check beta length and consistency
        if first_betas is None:
            first_betas = file_content["betas"]
        else:
            len_good = len(file_content["betas"]) == len(first_betas)
            if len_good and np.allclose(file_content["betas"], first_betas):
                continue
            print(f"Warning: beta doesn't agree for {filepath}")
            print(f"getting {file_content['betas']}")

    return pd.DataFrame.from_records(records_EoS)

def print_mu_vals_for_shell(mu_vals_joint, mu_vals_sub, SRoffset):
    all_sub_region_mus = np.round(np.concatenate([mu_vals_joint, mu_vals_joint - SRoffset]),3)
    all_mus_needed = np.setdiff1d(all_sub_region_mus, mu_vals_sub)
    if len(all_mus_needed) != 0:
        print("Run subregion simulations for the following mu:")
        print("(" + " ".join([f"{v:.4g}" for v in all_mus_needed]) + ")")

def filter_and_create_grid(
    df_all_data: pd.DataFrame,
    filter_params: dict,
    mu_col: str = None,
):
    """
    Filter DataFrame using specified parameters and unstack into 2D grids vs. (mu, beta).

    Parameters
    ----------
    df_all_data : pd.DataFrame
        DataFrame loaded via load_eos_data.
    filter_params : dict
        Parameters to filter by, e.g. {"V": 0.3, "shift": -2.0}.
    mu_col : str, optional
        Column name for chemical potential ("muR" for joint, "mu" for subregion).
        If None, automatically inferred.

    Returns
    -------
    mu_vals : np.ndarray
        Unique sorted chemical potential values.
    beta_vals : np.ndarray
        Unique sorted inverse temperature values.
    grids : dict of np.ndarray
        Dictionary of 2D grids with shape (len(mu_vals), len(beta_vals)).
        Keys include 'n_avg', 'S2', 'energy', 'eps', 'ov', and subregion quantities if available.
    df_filtered : pd.DataFrame
        The filtered DataFrame indexed by [mu_col, "beta"].
    """
    if mu_col is None:
        mu_col = "muR" if "muR" in df_all_data.columns else "mu"

    query_parts = []
    for k, v in filter_params.items():
        if k in df_all_data.columns:
            if isinstance(v, (int, float, np.floating, np.integer)):
                query_parts.append(f"({k} == {v})")
            else:
                query_parts.append(f"({k} == '{v}')")
    query_str = " & ".join(query_parts)

    df_filtered = df_all_data.query(query_str) if query_str else df_all_data.copy()
    idx_list = [mu_col, "beta"]
    df_filtered = df_filtered.set_index(idx_list).sort_index()

    mu_vals = df_filtered.index.get_level_values(mu_col).unique().to_numpy()
    beta_vals = df_filtered.index.get_level_values("beta").unique().to_numpy()

    grids = {}
    candidate_cols = [
        "n_avg",
        "S2",
        "energy",
        "eps",
        "ov",
        "n_avgS",
        "n_avgR",
        "S2S",
        "S2R",
        "energyS",
        "energyR",
    ]
    for col in candidate_cols:
        if col in df_filtered.columns:
            grids[col] = df_filtered[col].unstack("beta").loc[mu_vals, beta_vals].to_numpy()

    return mu_vals, beta_vals, grids, df_filtered


def compute_match_eos(
    mu_vals_target: np.ndarray,
    beta_vals_target: np.ndarray,
    mu_vals_sub: np.ndarray,
    beta_vals_sub: np.ndarray,
    grids_sub: dict,
    SRoffset: float,
) -> dict:
    """
    Calculate the composite 'match' EoS for the big system on the target (muR, beta) grid.

    The big system consists of two equal-sized subregions S and R:
    - Subregion S has chemical potential muS = muR - SRoffset
    - Subregion R has chemical potential muR

    The EoS for the combined system is obtained by interpolating subregion EoS
    along the mu axis at muS and muR, then averaging per-site quantities:
        n_match = (n_sub(muS) + n_sub(muR)) / 2
        S2_match = (S2_sub(muS) + S2_sub(muR)) / 2
        E_match = (E_sub(muS) + E_sub(muR)) / 2

    Parameters
    ----------
    mu_vals_target : np.ndarray
        Array of target muR values (e.g. from joint system).
    beta_vals_target : np.ndarray
        Array of target beta values.
    mu_vals_sub : np.ndarray
        Chemical potential values from the single-subregion calculation.
    beta_vals_sub : np.ndarray
        Inverse temperature values from the single-subregion calculation.
    grids_sub : dict
        Dictionary of single-subregion 2D grids (n_avg, S2, energy, etc.).
    SRoffset : float
        Chemical potential difference muR - muS.

    Returns
    -------
    grids_match : dict
        Dictionary of 2D grids on (mu_vals_target, beta_vals_target) for the match system.
    """
    assert np.all(np.diff(mu_vals_sub) > 0)
    N_mu = len(mu_vals_target)
    N_beta = len(beta_vals_target)
    grids_match = {}

    # Check if beta grids are identical
    betas_match = len(beta_vals_sub) == N_beta and np.allclose(beta_vals_sub, beta_vals_target)

    quantities = ["n_avg", "S2", "energy", "eps", "ov"]
    for q in quantities:
        if q not in grids_sub:
            continue

        val_sub = grids_sub[q]
        val_S = np.full((N_mu, N_beta), np.nan, dtype=float)
        val_R = np.full((N_mu, N_beta), np.nan, dtype=float)

        if betas_match:
            for j in range(N_beta):
                val_S[:, j] = np.interp(
                    mu_vals_target - SRoffset,
                    mu_vals_sub,
                    val_sub[:, j],
                    left=np.nan,
                    right=np.nan,
                )
                val_R[:, j] = np.interp(
                    mu_vals_target,
                    mu_vals_sub,
                    val_sub[:, j],
                    left=np.nan,
                    right=np.nan,
                )
        else:
            # Fallback 2D interpolation if beta grids differ
            print("beta grids differ - check the original data!")
            from scipy.interpolate import RegularGridInterpolator

            interp_fn = RegularGridInterpolator(
                (mu_vals_sub, beta_vals_sub),
                val_sub,
                bounds_error=False,
                fill_value=np.nan,
            )
            MU, BETA = np.meshgrid(mu_vals_target, beta_vals_target, indexing="ij")
            pts_S = np.stack([(MU - SRoffset).ravel(), BETA.ravel()], axis=-1)
            pts_R = np.stack([MU.ravel(), BETA.ravel()], axis=-1)
            val_S = interp_fn(pts_S).reshape(N_mu, N_beta)
            val_R = interp_fn(pts_R).reshape(N_mu, N_beta)

        # Average over the two subregions for whole-system per-site EoS
        grids_match[q] = 0.5 * (val_S + val_R)

        # Also store individual subregion contributions
        if q == "n_avg":
            grids_match["n_avgS"] = val_S
            grids_match["n_avgR"] = val_R
        elif q == "S2":
            grids_match["S2S"] = val_S
            grids_match["S2R"] = val_R
        elif q == "energy":
            grids_match["energyS"] = val_S
            grids_match["energyR"] = val_R

    return grids_match


def plot_eos_grids_2x2(
    beta_vals: np.ndarray,
    mu_vals: np.ndarray,
    grids: dict,
    title: str = None,
    mu_label: str = r"$\mu$",
):
    """
    Plot 2x2 grid of thermodynamic quantities: n_avg, S2_avg, Energy, and max infidelity.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    plot_items = [
        (grids["n_avg"],    None,       r"$n_{\rm avg}$"),
        (grids["S2"],       None,       r"$S_{2, \rm avg}$"),
        (grids["energy"],   None,       r"$E$"),
        (1.0 - grids["ov"], (0, 0.01),  "max infidelity (1 - ov)"),
    ]

    for ax, (vals, clim, str_label) in zip(axes.flatten(), plot_items):
        if vals is None:
            continue
        img = ax.pcolormesh(beta_vals, mu_vals, vals, clim=clim, shading="auto")
        ax.set_box_aspect(1)
        ax.set_title(str_label)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(mu_label)
        fig.colorbar(img, ax=ax)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

def plot_eos_grids_2x3(
    beta_vals: np.ndarray,
    mu_vals: np.ndarray,
    grids: dict,
    title: str = None,
    mu_label: str = r"$\mu$",
):
    """
    Plot 2x2 grid of thermodynamic quantities: n_avg, S2_avg, Energy, and max infidelity.
    """
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7))
    plot_items = [
        (grids["n_avg"],    None,       r"$n_{\rm avg}$"),
        (grids["S2"],       None,       r"$S_{2, \rm avg}$"),
        (grids["n_avgS"],   None,       r"$n_{\rm S}$"),
        (grids["energy"],   None,       r"$E$"),
        (1.0 - grids["ov"], (0, 0.01),  "max infidelity (1 - ov)"),
        (grids["S2S"],      None,       r"$S_{\rm S}$"),
    ]

    for ax, (vals, clim, str_label) in zip(axes.flatten(), plot_items):
        if vals is None:
            continue
        img = ax.pcolormesh(beta_vals, mu_vals, vals, clim=clim, shading="auto")
        ax.set_box_aspect(1)
        ax.set_title(str_label)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(mu_label)
        fig.colorbar(img, ax=ax)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

def plot_eos_difference_2x2(
    beta_vals: np.ndarray,
    mu_vals: np.ndarray,
    grids_joint: dict,
    grids_match: dict,
    title: str = None,
    mu_label: str = r"$\mu_R$",
    SRoffset: float = None,
):
    """
    Plot 2D heatmaps of the difference between Joint and Match EoS:
    Delta n_avg, Delta S2_avg, Delta Energy, and relative difference Delta n / n.
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    diff_n = grids_joint["n_avg"] - grids_match["n_avg"]
    diff_s2 = grids_joint["S2"] - grids_match["S2"]
    diff_e = grids_joint["energy"] - grids_match["energy"]
    diff_nS = grids_joint["n_avgS"] - grids_match["n_avgS"]
    diff_s2S = grids_joint["S2S"] - grids_match["S2S"]
    # Relative density difference
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff_n = diff_n / grids_joint["n_avg"]
        rel_diff_nS = diff_nS / grids_joint["n_avgS"]
        rel_diff_S2S = diff_s2S / grids_joint["S2S"]

    plot_items = [
        (diff_n,       None,        r"$\Delta n = n_{\rm joint} - n_{\rm match}$"),
        (diff_s2,      None,        r"$\Delta S_{2} = S_{2, \rm joint} - S_{2, \rm match}$"),
        (diff_nS,      None,        r"$\Delta n_{\rm S} = n_{\rm S, joint} - n_{\rm S, match}$"),
        (rel_diff_nS,  (-0.5, 0.5), r"$\Delta n_{\rm S} / n_{\rm S, joint}$ (relative)"),
        (diff_e,       None,        r"$\Delta E = E_{\rm joint} - E_{\rm match}$"),
        (rel_diff_n,   (-0.1, 0.1), r"$\Delta n / n_{\rm joint}$ (relative)"),
        (diff_s2S,     None,        r"$\Delta S_{\rm S} = S_{\rm S, joint} - S_{\rm S, match}$"),
        (rel_diff_S2S, (-0.5, 0.5), r"$\Delta S_{\rm S} / S_{\rm S, joint}$ (relative)"),
    ]

    for ax, (vals, clim, str_label) in zip(axes.flatten(), plot_items):
        max_abs = np.nanmax(np.abs(vals))
        if clim is None: clim = (-max_abs, max_abs)
        img = ax.pcolormesh(beta_vals, mu_vals, vals, clim=clim, cmap="coolwarm", shading="auto")
        ax.set_box_aspect(1)
        ax.set_title(str_label)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(mu_label)
        if SRoffset:
            ryax = ax.secondary_yaxis('right', functions = (lambda x: x - SRoffset, lambda x: x + SRoffset))
            ryax.set_ylabel(r"$\mu_S$")
        fig.colorbar(img, ax=ax)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_eos_1d_cuts(
    beta_vals: np.ndarray,
    mu_vals: np.ndarray,
    grids_joint: dict,
    grids_match: dict,
    selected_betas: list = None,
    title: str = None,
):
    """
    Plot 1D cuts comparing Joint (solid) vs. Match (dashed) as a function of muR
    at selected inverse temperatures beta.
    """
    if selected_betas is None:
        selected_betas = [0.5, 2.0, 5.0, 10.0, 20.0]

    # Find closest available beta indices
    beta_indices = []
    actual_betas = []
    for b in selected_betas:
        idx = int(np.argmin(np.abs(beta_vals - b)))
        if idx not in beta_indices:
            beta_indices.append(idx)
            actual_betas.append(beta_vals[idx])

    # diff_n = grids_joint["n_avg"] - grids_match["n_avg"]
    # diff_s2 = grids_joint["S2"] - grids_match["S2"]
    # diff_e = grids_joint["energy"] - grids_match["energy"]
    diff_nS = grids_joint["n_avgS"] - grids_match["n_avgS"]
    diff_s2S = grids_joint["S2S"] - grids_match["S2S"]
    diff_eS = grids_joint["energyS"] - grids_match["energyS"]
    # Relative density difference
    with np.errstate(divide="ignore", invalid="ignore"):
        # rel_diff_n = diff_n / grids_joint["n_avg"]
        rel_diff_nS = diff_nS / grids_joint["n_avgS"]
        rel_diff_S2S = diff_s2S / grids_joint["S2S"]
        rel_diff_eS = diff_eS / grids_joint["n_avg"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    quantities = [
        ("n_avg",   axes[0, 0], r"$n_{\rm avg}$"),
        ("S2",      axes[0, 1], r"$S_{2, \rm avg}$"),
        ("energy",  axes[0, 2], r"$E$"),
        ("n_avgS",  axes[1, 0], r"$n_{\rm S}$"),
        ("S2S",     axes[1, 1], r"$S_{2, \rm S}$"),
        ("energyS", axes[1, 2], r"$E_{\rm S}$"),
    ]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(actual_betas)))

    for q_key, ax, q_label in quantities:
        for idx, b_val, c in zip(beta_indices, actual_betas, colors):
            yj = grids_joint[q_key][:, idx]
            ym = grids_match[q_key][:, idx]
            ax.plot(mu_vals, yj, color=c, linestyle="-", label=rf"$\beta={b_val:.1f}$ (joint)")
            ax.plot(mu_vals, ym, color=c, linestyle="--", label=rf"$\beta={b_val:.1f}$ (match)")
            axes[2, 0].plot(mu_vals, rel_diff_nS[:, idx], color=c, linestyle="-")
            axes[2, 1].plot(mu_vals, rel_diff_S2S[:, idx], color=c, linestyle="-")
            axes[2, 2].plot(mu_vals, rel_diff_eS[:, idx], color=c, linestyle="-")

        ax.set_xlabel(r"$\mu_R$")
        ax.set_ylabel(q_label)
        ax.set_title(q_label)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Place legend on the first subplot
    axes[0, 0].legend(fontsize=7, ncol=2)

    # Limits for the relative value plots
    axes[2, 0].set_ylim(-0.2, 0.2)
    axes[2, 1].set_ylim(-0.5, 0.5)
    axes[2, 2].set_ylim(-0.01, 0.2)
    axes[2, 0].set_title(r"$\Delta n_{\rm S} / n_{\rm S, joint}$ (relative)")
    axes[2, 1].set_title(r"$\Delta S_{\rm S} / S_{\rm S, joint}$ (relative)")
    axes[2, 2].set_title(r"$\Delta E_{\rm S} / E_{\rm S, joint}$ (relative)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


#%% Main execution script
if __name__ == "__main__":
    # =========================================================================
    # 1. Shared simulation parameters
    # =========================================================================
    data_folder = Path(r"C:\Users\ken92\Downloads\Shao-Wen Data")
    geometry = "sawtooth"  # "sawtooth", "kagome"
    chi_tenpy = 64
    V_model = 0.3
    tp_ST = 1.41421356237
    shift_ST = -2.0

    # Subregion parameters (User specified)
    SRoffset = 2.0         # muR - muS
    L_subregion = 50       # Number of unit cells in each subregion

    # Joint system (Derived from subregion parameters)
    L_joint = 100          # Number of unit cells in the joint system (2 * L_subregion)

    # Shared derived parameters
    E_flat_ST = 2.0 + shift_ST   # Flat band energy with shift (noninteracting)
    E_min = min(E_flat_ST, -4.0)
    E_max = max(E_flat_ST, 0.0)
    geom_spec_str = f"V{V_model}_tp{tp_ST}_shift{shift_ST}"
    filter_params = {"V": V_model, "shift": shift_ST, "tp": tp_ST}

    print("=" * 70)
    print("EoS Joint vs. Match Comparison")
    print(f"Geometry: {geometry}, {geom_spec_str}, chi = {chi_tenpy}")
    print(f"L_subregion = {L_subregion}, L_joint = {L_joint}, SRoffset = {SRoffset}")
    print("=" * 70)

    # =========================================================================
    # 2. Joint system: Load, filter, and extract EoS grids
    # =========================================================================
    print("\n--- Loading Joint System Data ---")
    scan_name_joint = (
        f"ST_SL_I_L{L_joint}_t1_tpsqrt2_shf{int(shift_ST)}_V{V_model}"
        f"_chi{chi_tenpy}_SRoff{np.round(SRoffset, 1)}"
    )
    scan_folder_joint = Path(data_folder, scan_name_joint)
    assert scan_folder_joint.exists(), f"Joint folder does not exist: {scan_folder_joint}"

    tnpspc_joint = {
        "geometry": geometry,
        "L": L_joint,
        "chi": chi_tenpy,
    }

    df_all_joint = load_eos_data(scan_folder_joint, tnpspc_joint, is_joint=True)
    mu_vals_joint, beta_vals_joint, grids_joint, df_filtered_joint = filter_and_create_grid(
        df_all_joint, filter_params, mu_col="muR"
    )

    print(f"Joint grid: {len(mu_vals_joint)} muR points x {len(beta_vals_joint)} beta points")
    print(f"muR range: [{mu_vals_joint.min():.2f}, {mu_vals_joint.max():.2f}]")
    print(f"beta range: [{beta_vals_joint.min():.2f}, {beta_vals_joint.max():.2f}]")

    # Plot joint EoS
    fig_joint, axes_joint = plot_eos_grids_2x3(
        beta_vals_joint,
        mu_vals_joint,
        grids_joint,
        title=f"Joint system: {geometry} lattice, L={L_joint}, {geom_spec_str}, SRoffset={SRoffset}",
        mu_label=r"$\mu_R$",
    )

    # =========================================================================
    # 3. Subregion (single) system: Load, filter, and extract EoS grids
    # =========================================================================
    print("\n--- Loading Subregion (Single) System Data ---")
    scan_name_sub = (
        f"ST_SL_I_L{L_subregion}_t1_tpsqrt2_shf{int(shift_ST)}_V{V_model}_chi{chi_tenpy}"
    )
    scan_folder_sub = [d for d in data_folder.glob(f"{scan_name_sub}*")]
    # scan_folder_sub = Path(data_folder, scan_name_sub)
    # if not scan_folder_sub.exists():
    #     # Check fallback with .tar suffix if folder was created with .tar in name
    #     scan_folder_sub = Path(data_folder, scan_name_sub + ".tar")
    # assert scan_folder_sub.exists(), f"Subregion folder does not exist: {scan_folder_sub}"

    tnpspc_sub = {
        "geometry": geometry,
        "L": L_subregion,
        "chi": chi_tenpy,
    }

    df_all_sub = load_eos_data(scan_folder_sub, tnpspc_sub, is_joint=False)
    mu_vals_sub, beta_vals_sub, grids_sub, df_filtered_sub = filter_and_create_grid(
        df_all_sub, filter_params, mu_col="mu"
    )

    print(f"Subregion grid: {len(mu_vals_sub)} mu points x {len(beta_vals_sub)} beta points")
    print(f"Subregion mu range: [{mu_vals_sub.min():.2f}, {mu_vals_sub.max():.2f}]")
    print(f"Subregion beta range: [{beta_vals_sub.min():.2f}, {beta_vals_sub.max():.2f}]")
    print_mu_vals_for_shell(mu_vals_joint, mu_vals_sub, SRoffset)

    # Plot subregion EoS
    fig_sub, axes_sub = plot_eos_grids_2x2(
        beta_vals_sub,
        mu_vals_sub,
        grids_sub,
        title=f"Single subregion: {geometry} lattice, L={L_subregion}, {geom_spec_str}",
        mu_label=r"$\mu$",
    )

    # =========================================================================
    # 4. Construct Match EoS (interpolating and summing the two subregions)
    # =========================================================================
    print("\n--- Computing Match EoS (Sum of two subregions with SRoffset) ---")
    grids_match = compute_match_eos(
        mu_vals_target=mu_vals_joint,
        beta_vals_target=beta_vals_joint,
        mu_vals_sub=mu_vals_sub,
        beta_vals_sub=beta_vals_sub,
        grids_sub=grids_sub,
        SRoffset=SRoffset,
    )

    # Plot match EoS
    fig_match, axes_match = plot_eos_grids_2x3(
        beta_vals_joint,
        mu_vals_joint,
        grids_match,
        title=f"Match system: 2 x L={L_subregion} combined, {geom_spec_str}, SRoffset={SRoffset}",
        mu_label=r"$\mu_R$",
    )
    axes_match[1, 1].set_title("(infidelity doesn't necessarily make sense)")

    # =========================================================================
    # 5. Compare Joint vs. Match EoS
    # =========================================================================
    print("\n--- Comparing Joint vs. Match EoS ---")
    diff_n = grids_joint["n_avg"] - grids_match["n_avg"]
    diff_s2 = grids_joint["S2"] - grids_match["S2"]
    diff_e = grids_joint["energy"] - grids_match["energy"]

    print("Summary of differences (Joint - Match):")
    print(f"  Particle density n:  Max abs = {np.nanmax(np.abs(diff_n)):.6f},  Mean abs = {np.nanmean(np.abs(diff_n)):.6f}")
    print(f"  Renyi-2 entropy S2:  Max abs = {np.nanmax(np.abs(diff_s2)):.6f},  Mean abs = {np.nanmean(np.abs(diff_s2)):.6f}")
    print(f"  Energy per site E:   Max abs = {np.nanmax(np.abs(diff_e)):.6f},  Mean abs = {np.nanmean(np.abs(diff_e)):.6f}")

    # Subregion level comparisons if available
    if "n_avgS" in grids_joint and "n_avgS" in grids_match:
        diff_n_S = grids_joint["n_avgS"] - grids_match["n_avgS"]
        diff_n_R = grids_joint["n_avgR"] - grids_match["n_avgR"]
        print(f"  Subregion S density: Max abs = {np.nanmax(np.abs(diff_n_S)):.6f},  Mean abs = {np.nanmean(np.abs(diff_n_S)):.6f}")
        print(f"  Subregion R density: Max abs = {np.nanmax(np.abs(diff_n_R)):.6f},  Mean abs = {np.nanmean(np.abs(diff_n_R)):.6f}")

    # 2D difference heatmaps
    fig_diff, axes_diff = plot_eos_difference_2x2(
        beta_vals_joint,
        mu_vals_joint,
        grids_joint,
        grids_match,
        title=f"Difference (Joint - Match): {geometry} lattice, {geom_spec_str}, SRoffset={SRoffset}",
        mu_label=r"$\mu_R$",
        SRoffset = SRoffset
    )

    # 1D slice comparison cuts across muR at representative beta values
    fig_cuts, axes_cuts = plot_eos_1d_cuts(
        beta_vals_joint,
        mu_vals_joint,
        grids_joint,
        grids_match,
        selected_betas=[0.5, 2.0, 5.0, 10.0, 20.0],
        title=f"1D Cuts vs. $\\mu_R$: Joint (solid) vs. Match (dashed) at fixed $\\beta$",
    )

    # =========================================================================
    # 6. Additional plot stuff
    # =========================================================================
    # eyeballed reservoir flat band energy
    try:
        if V_model == 0.3:
            E_flat_R = 0.5
        elif V_model == 0.1:
            E_flat_R = 0.2
        else:
            raise(ValueError(f"flat band energy for V = {V_model} not eyeballed yet"))
        E_flat_S = E_flat_R + SRoffset
        for axes in [axes_joint, axes_sub, axes_match, axes_diff]:
            for ax in axes.flatten():
                ax.axhline(E_flat_R, color = "r", ls = "--", lw = 0.7, label = r"$E_{\mathrm{flat, R}}$")
                ax.axhline(E_flat_S, color = "c", ls = "--", lw = 0.7, label = r"$E_{\mathrm{flat, S}}$")
            axes[0, 0].legend()
        for ax in axes_cuts.flatten():
            ax.axvline(E_flat_R, color = "r", ls = "--", lw = 0.7, label = r"$E_{\mathrm{flat, R}}$")
            ax.axvline(E_flat_S, color = "c", ls = "--", lw = 0.7, label = r"$E_{\mathrm{flat, S}}$")
    except ValueError as e:
        print(e)

    print("\nAll figures generated successfully. Displaying plots...")
    plt.show()
