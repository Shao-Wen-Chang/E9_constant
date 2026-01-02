# %%
import sys
from pathlib import Path
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if str(E9path) not in sys.path:
    sys.path.insert(1, str(E9path))
from E9_fn import util

import util_match
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# %%
bool_plot = False   # set to false if just gathering matching data

# %% [markdown]
# # Reading data
# In case that we get different EoS using the same folder parameters in the future, I'll also keep track of the EoS file. For this I think the file modification time is sufficient.
# 
# Unfinished runs are detected by mismatches in the length of beta.
# 
# How I treat folder parameters:
# * (sawtooth) $\mu$, $V$, $t_p$ and $shift$ / (kagome) $W$ are things we expect to be different in a folder. All the other parameters should be the same; I include them in `folder_params_shared` and check for every folder.
#     * Among them, $\mu$ is additionally saved in `data_in_measurements["mu"]`. I check that this indeed matches the one specified in the folder name.
#     * For $V$, $t_p$ and $shift$, I don't do the filtering until all the data are loaded.
# * I assume $t = 1$.
# * There is also a file with `dBeta' = -dBeta` that goes to negative temperatures. I remove the `beta = 0` row from this one and piece the two files together.
# * Note that $s$ is the entropy per site, not per particle. I will use $\sigma_s$ (`sigma_s`) and $\sigma_r$ (`sigma_r`) for entropy per particle. $\sigma$ seems to be the right quantity to look at to evaluate cooling (or at least is what e.g. Greiner's people always look at).

# %%
# I had to keep the data files in the download folder, otherwise the folder names are too long
data_folder = Path(r"C:\Users\ken92\Downloads\Shao-Wen Data")

# tenpy specs used to generate folder names
geometry = "kagome" # "sawtooth", "kagome"
L_subregion = 50
mumin = -10.0
mumax = 10.0
dMu = 0.01
dBeta = 0.1           # Use positive here; minus sign added by hand below

# geometry specific parameters
ymax_ST = None
tp_ST = 1.41421356237
shift_ST = -2.0
W_kagome = 4    # width of the kagome strip, only used if geometry = "kagome"

# %% [markdown]
# Some geometry dependent variables.

# %%
file_prefix = ""        # File name for Sajant's data
geom_spec_str = ""      # For plotting etc
N_sites = 0             # Number of sites in each region
E_min, E_max = 0., 0.   # Lowest / highest band energies
filter_params = dict()  # Same name as eos_matcher for interacting system

if geometry == "sawtooth":
    file_prefix = (f"{geometry}_spinless_NI_L{L_subregion}_t1.0_tp{tp_ST}_shift{shift_ST}"
                    f"_ymax{ymax_ST}_mumin{mumin}_mumax{mumax}_dMu{dMu}")
    geom_spec_str = f"tp{tp_ST}_shift{shift_ST}"
    N_sites = 2 * L_subregion
    E_flat_ST = 2. + shift_ST   # energy of the flat band with shift (assuming ymax is None)
    E_min = min(E_flat_ST, -4.)
    E_max = max(E_flat_ST, 0.)
    filter_params = {
        "tp": tp_ST,
        "shift": shift_ST,
        "ymax": None,
    }
elif geometry == "kagome":
    file_prefix = (f"{geometry}_spinless_NI_L{L_subregion}_W{W_kagome}_t1.0_"
                    f"_mumin{mumin}_mumax{mumax}_dMu{dMu}")
    geom_spec_str = f"W{W_kagome}"
    N_sites = 3 * W_kagome * L_subregion
    E_min = -4.
    E_max = 2.
    filter_params = {
        "W": W_kagome,
    }

# %%
# positive beta
scan_name = file_prefix + f"_dBeta{dBeta}_Nsteps1000.pkl"
# scan_name = file_prefix + f"_dBeta{dBeta}_Nsteps200.pkl"
filepath = Path(data_folder, scan_name)
assert filepath.is_file()
with open(filepath, "rb") as f:
    file_content = pickle.load(f)

# negative beta
scan_name_neg = file_prefix + f"_dBeta{-dBeta}_Nsteps200.pkl"
filepath_neg = Path(data_folder, scan_name_neg)
assert filepath_neg.is_file()
with open(filepath_neg, "rb") as f:
    file_content_neg = pickle.load(f)

# %% [markdown]
# ## Plots
# The sharp feature in $S_2$ plot is the flat band.
# * Sawtooth: there's another weaker feature at around $\mu = -4$, which should be the band minimum.
# * Kagome: there's a strong qualitative dependence on $W$ in each plot, since it determines the number of sites in each unit cell.

# %%
# index is xxx_vals[i_mu, i_beta]
mu_vals = file_content["mus"]
beta_vals_pos = file_content["betas"]
sVN_grid_pos = file_content["data"][:,:,0].T  # Von Neumann entropy per site
s2_grid_pos = file_content["data"][:,:,1].T   # Renyi 2-entropy per site
n_grid_pos = file_content["data"][:,:,2].T    # particle per site
e_grid_pos = file_content["data"][:,:,3].T    # energy per site

assert np.allclose(mu_vals, file_content_neg["mus"])
beta_vals_neg = file_content_neg["betas"]
sVN_grid_neg = file_content_neg["data"][:,:,0].T  # Von Neumann entropy per site
s2_grid_neg = file_content_neg["data"][:,:,1].T   # Renyi 2-entropy per site
n_grid_neg = file_content_neg["data"][:,:,2].T    # particle per site
e_grid_neg = file_content_neg["data"][:,:,3].T    # energy per site

# %%
# sort in increasing beta
beta_vals = np.concatenate([beta_vals_neg[:0:-1], beta_vals_pos])
sVN_grid = np.concatenate([sVN_grid_neg[:,:0:-1], sVN_grid_pos], axis = 1)  # Von Neumann entropy per site
s2_grid = np.concatenate([s2_grid_neg[:,:0:-1], s2_grid_pos], axis = 1)   # Renyi 2-entropy per site
n_grid = np.concatenate([n_grid_neg[:,:0:-1], n_grid_pos], axis = 1)    # particle per site
e_grid = np.concatenate([e_grid_neg[:,:0:-1], e_grid_pos], axis = 1)    # energy per site

N_betas = len(beta_vals)
N_mus = len(mu_vals)

# %%
print(n_grid.shape)
print(mu_vals.shape)
print(beta_vals.shape)

# %%
if bool_plot:
    fig_ns, axes_ns = plt.subplots(2, 2, figsize = (8, 7))
    for ax, vals, str_label, clim, levels in zip(axes_ns.flatten(),
                        [n_grid, s2_grid, e_grid, sVN_grid],
                        ["n_avg", "S2_avg", "E", "sVN_svg"],
                        [None, None, None, None],
                        [[1e-4, 1e-2, 0.1, 0.3, 0.99],
                        [1e-4, 1e-2, 0.1, 0.3],
                        [-1, -0.99, -0.5, -0.1],
                        [1e-4, 1e-2, 0.1, 0.3]]):
        img = ax.pcolormesh(beta_vals, mu_vals, vals, clim = clim)
        ax.set_box_aspect(1)
        ax.set_title(str_label)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\mu$")
        # cntr = ax.contour(beta_vals, mu_vals, vals, colors = "white", linestyles = "solid", levels = levels)
        # ax.clabel(cntr, inline = True)
        fig_ns.colorbar(img, ax = ax)
    fig_ns.suptitle(f"{geometry} lattice, {geom_spec_str}")
    fig_ns.tight_layout()

# %% [markdown]
# Zoom in to lower $\beta$ (higher $T$).

# %%
mask_beta_zoom = np.logical_and(np.greater_equal(beta_vals, -5), np.less_equal(beta_vals, 5))
if bool_plot:
    fig_ns_zoom, axes_ns_zoom = plt.subplots(2, 2, figsize = (8, 7))
    for ax, vals, str_label, clim, levels in zip(axes_ns_zoom.flatten(),
                        [n_grid[:,mask_beta_zoom], s2_grid[:,mask_beta_zoom], e_grid[:,mask_beta_zoom], sVN_grid[:,mask_beta_zoom]],
                        ["n_avg", "S2_avg", "E", "sVN_svg"],
                        [None, (0, 0.5), None, None],
                        [[1e-4, 1e-2, 0.1, 0.3, 0.6, 0.99],
                        [1e-4, 1e-2, 0.05, 0.1, 0.3],
                        [-1, -0.99, -0.5, -0.1],
                        [1e-4, 1e-2, 0.1, 0.3]]):
        img = ax.pcolormesh(beta_vals[mask_beta_zoom], mu_vals, vals, clim = clim)
        ax.set_box_aspect(1)
        ax.set_title(str_label)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\mu$")
        # cntr = ax.contour(beta_vals[mask_beta_zoom], mu_vals, vals, colors = "white", linestyles = "solid", levels = levels)
        # ax.clabel(cntr, inline = True)
        fig_ns_zoom.colorbar(img, ax = ax)
    fig_ns_zoom.suptitle(f"{geometry} lattice, {geom_spec_str}")
    fig_ns_zoom.tight_layout()

# %% [markdown]
# Look at the value of one parameter (e.g. $s$) holding something else (e.g. $n$) fixed.

# %%
# beta_curve, s2_curve = util.parametric_slice_2D(mu_vals, beta_vals, s2_grid, n_grid, 0.4,
#                                                 single_valued = "axis1")
# plt.plot(mu_vals, s2_curve)

# %% [markdown]
# # Actual calculations
# Goal: given our equations of state $n(\mu, \beta)$ and $s_2(\mu, \beta)$, achieve some filling factor $n_s$ in the system for different $(N_{tot}, S_{tot})$
# (This also fixes $N_r = N_{tot} - n_s N_{site}$, or equivalently $n_r = 2n - n_s$)
# 1. Find $\mu_{s, r} = \mu(\beta) |_{n(\mu, \beta) = n_{s, r}}$ by inverting $n(\mu, \beta)$. I can do this for all $\beta$.
# 2. Find $s_2(\mu_{s, r}, \beta)$. As I vary $\beta$ I should get a smooth change in $S_2 = N_{site} (s_2(\mu_s) + s_2(\mu_r))$, so the scan in $S_{tot}$ comes for free.
# 3. Do this for different $N_{tot}$. Now I've in principle done a scan in the $(N, S)$ space.
# 
# (Take $L_s = L_r = L$ for now, where $L$ is the number of unit cells)
# 
# Here we get $(n_s; N_{tot}, \beta) \rightarrow (\mu_s, \mu_r, S_{tot})$, where I additionally define
# * $V_{offset} = \mu_r - \mu_s$ is the offset potential required to achieve $n_s$.
# * $\mu \equiv \mu_r$ --- this one is somewhat arbitrary. Effectively, I'm assuming that $V_{offset}$ is a potential that only increases the potential energy in the system region, while leaving the reservoir region intact. This is the case for our experimental setup, since our DMD light is blue-detuned.
# * $n = N_{tot} / 2 N_{site}$
# 
# I write a semicolon for $n_s$ because it is not really a thermodynamic variable in the usual sense, but not including that gives one the wrong impression that the relation is underdetermined. (i.e. I want to have the same number of values on both side)
# 
# ## $N_{tot}$ and $S_{tot}$ matching
# We'll see how well ChatGPT implement this algorithm in 48 seconds.

# %% [markdown]
# ### Scan $N_{tot}$ (jump here to change $n_s$)

# %%
# idx_beta_scan = np.logical_and(beta_vals >= -20., beta_vals <= 20.)
idx_beta_scan = (beta_vals >= -100.)      # All beta, use this for the .csv file
beta_scan = beta_vals[idx_beta_scan]
for n_s_tar in np.arange(0.05, 0.95 + 0.001, step = 0.02):
    N_tot_arr = np.arange((n_s_tar + 0.) * N_sites, (n_s_tar + 1) * N_sites + 1, dtype=int)

    # %%
    N_s_tar = n_s_tar * N_sites
    assert N_s_tar <= N_tot_arr[-1], f"there must be at least {N_s_tar} particles"

    # %%
    N_beta = beta_scan.size
    N_Ntot = N_tot_arr.size

    mu_s_all  = np.full((N_Ntot, N_beta), np.nan)
    mu_r_all  = np.full((N_Ntot, N_beta), np.nan)
    s2_s_all  = np.full((N_Ntot, N_beta), np.nan)
    s2_r_all  = np.full((N_Ntot, N_beta), np.nan)
    S2_tot_all = np.full((N_Ntot, N_beta), np.nan)
    n_r_arr   = np.full(N_Ntot, np.nan)

    for k, N_tot in enumerate(N_tot_arr):
        # global filling n and reservoir filling n_r
        n_global = N_tot / N_sites / 2.
        n_r = 2.0 * n_global - n_s_tar
        n_r_arr[k] = n_r

        # 1) invert n(mu, beta) to get mu_s(beta) and mu_r(beta)
        mu_s = util_match.invert_n_to_mu(mu_vals, beta_scan, n_grid[:, idx_beta_scan], n_s_tar)
        mu_r = util_match.invert_n_to_mu(mu_vals, beta_scan, n_grid[:, idx_beta_scan], n_r)

        mu_s_all[k, :] = mu_s
        mu_r_all[k, :] = mu_r

        # 2) evaluate s2 at those mus
        s2_s = util_match.s2_at_mu(mu_vals, beta_scan, s2_grid[:, idx_beta_scan], mu_s)
        s2_r = util_match.s2_at_mu(mu_vals, beta_scan, s2_grid[:, idx_beta_scan], mu_r)

        s2_s_all[k, :] = s2_s
        s2_r_all[k, :] = s2_r

        # 3) total S2
        S2_tot_all[k, :] = N_sites * (s2_s + s2_r)

    dmu_all = mu_r_all - mu_s_all
    mu_glob_all = mu_r_all
    sigma2_s_all = s2_s_all / n_s_tar # entropy per particle
    sigma2_r_all = s2_r_all / n_r_arr[:, np.newaxis]
    sigma2_all = S2_tot_all / 2 / N_sites

    # %%
    for label, data in [("mu_s", mu_s_all), ("mu_r", mu_r_all), ("s2_s", s2_s_all), ("s2_r", s2_r_all)]:
        if np.all(np.isnan(data)):
            raise(ValueError(f"No legal values for {label} within the range of input"))

    # %% [markdown]
    # We are interested in knowing when we can realize $\sigma_r > \sigma_s$. I'll define a dimensionless figure of merit
    # 
    # $\varsigma \equiv \sigma_s / \sigma - 1$
    # 
    # that measures the decrease in entropy, measured in entropy per particle if there were no offset. $\varsigma < 0$ indicates cooling effect, and $\varsigma = -1$ means zero entropy for system particles.

    # %%
    varsigma_all = sigma2_s_all / sigma2_all - 1
    min_varsigma = np.nanmin(varsigma_all.flatten())    # best cooling obtained

    # masking arrays derived from varsigma
    varsigma_all_nonan = np.nan_to_num(varsigma_all)
    mask_alpha_cooling = np.minimum(varsigma_all_nonan, 0.) / min_varsigma

    # %%
    min_varsigma

    # %% [markdown]
    # ### Plots
    # The amount of cooling is, of course, shaded in red, in each axes. The last axes is a standalone visualization for it that has a fixed color range.
    # * $s_s$ plot is by definition boring, since $N_{tot}$ doesn't change $n_s$.
    # * $s_r$ plot has more feature because $n_r$ depends on $N_{tot}$.
    # * $S_{tot}$
    # * $\mu$
    # * $V_{offset}$ tells us the required offset we need to achieve $n_s$ in the system.
    # * $\varsigma$: the red grids in other axeses are scaled to the maximum amount of cooling. This one always use the same scale for comparison between different input parameters.

    # %%
    if bool_plot:
        fig_eos, axes_eos_all = plt.subplots(2, 3 * 2, figsize = (13, 7), gridspec_kw = {"width_ratios": [15, 1] * 3})
        axes_eos = axes_eos_all[:, ::2]
        axes_eos_cb = axes_eos_all[:,1::2]
        alpha_cooling_max = 0.
        for ax, ax_cb, vals, cmap, str_label, clim in zip(axes_eos.flatten(), axes_eos_cb.flatten(),
                [s2_s_all, s2_r_all, S2_tot_all, mu_glob_all, dmu_all, varsigma_all],
                ["viridis", "viridis", "viridis", "viridis", "viridis", "coolwarm"],
                [r"$s_s$", r"$s_r$", r"$S_{tot}$", r"$\mu$", r"$V_{offset}$", r"$\varsigma$"],
                [None, None, None, (E_min - 2, E_max + 2), (E_min - 2, E_max + 2), (-1, 4)]):
            img = ax.pcolormesh(beta_scan, N_tot_arr, vals, clim = clim, cmap = cmap)
            ax.set_box_aspect(1)
            ax.set_title(str_label)
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel(r"$N_{tot}$")
            ryax = ax.secondary_yaxis('right', functions = (lambda x: (x - N_s_tar) / (N_sites), lambda x: x * N_sites + N_s_tar))
            ryax.set_ylabel(r"$n_r$")
            # cntr = ax.contour(beta_scan, N_tot_arr, vals, colors = "white", linestyles = "solid", levels = 7)
            # ax.clabel(cntr, inline = True)
            fig_eos.colorbar(img, cax = ax_cb, fraction = 0.5, location = 'right')
            if str_label != r"$\varsigma$" and not np.isclose(alpha_cooling_max, 0.):
                ax.pcolormesh(beta_scan, N_tot_arr, np.ones_like(vals), clim = (0, 1), color = "red",# lw = 0.,
                            alpha = mask_alpha_cooling * alpha_cooling_max)
        fig_eos.suptitle(rf"results from EoS matching - $n_s =${n_s_tar} ($N_s =${N_s_tar}), {geometry} lattice, {geom_spec_str}")
        fig_eos.tight_layout(w_pad = 0.1)

    # %% [markdown]
    # Now, we scanned $\beta$ because it is easier to do, but we will be more interested in scanning total entropy. This is done in `matching_result_inverter.ipynb`.
    # 
    # ### Building the matching results Dataframe

    # %%
    records_matching = []
    for i_beta, beta in enumerate(beta_scan):
        for i_Ntot, Ntot in enumerate(N_tot_arr):
            records_matching.append({
                "beta": beta,
                "Ntot": int(Ntot),
                "n_s": np.round(n_s_tar, 4),
                "n_r": n_r_arr[i_Ntot],
                "mu_s": mu_s_all[i_Ntot, i_beta],
                "mu_r": mu_r_all[i_Ntot, i_beta],
                "V_offset": dmu_all[i_Ntot, i_beta],
                "mu_glob": mu_glob_all[i_Ntot, i_beta],
                "s2_s": s2_s_all[i_Ntot, i_beta],
                "s2_r": s2_r_all[i_Ntot, i_beta],
                "sigma2_s": sigma2_s_all[i_Ntot, i_beta],
                "sigma2_r": sigma2_r_all[i_Ntot, i_beta],
                "sigma2": sigma2_all[i_Ntot, i_beta],
                "S2_tot": S2_tot_all[i_Ntot, i_beta],
                "varsigma": varsigma_all[i_Ntot, i_beta],
            } | filter_params)

    # %%
    df_matching_rslt = pd.DataFrame.from_records(records_matching)
    # df_matching_rslt

    # %% [markdown]
    # ### Save to .csv
    # I'll treat the following as input parameters:
    # * All the original folder parameters except for $\mu$.
    # * $\beta$, $N_{tot}$, $n_s$
    # * EoS file modification time
    # 
    # A new row of data is added for each input parameters. If an entry with the same input parameters already exists in `csv_path`, we compare if the obtained results agree with the existing ones. If not, we print a warning but leaves the original result unchanged and move on.
    # 
    # Saving everything to a single .csv makes the comparison very slow (several minutes), so I make a new .csv for each $V$, $t_p$, $shift$, and $n_s$. For parameters in the name of .csv, the parameter comparison is done when looking for an existing `csv_path`.
    # 
    # If all the input parameters are the same except for the file modification time, the original entry is replaced if `bool_overwrite_older_mtime = True`, or a new row is added if it is `False`.

    # %%
    # raise(Exception("Imma stop here"))  # comment this out to actually save data

    # %% [markdown]
    # Check the number of rows existed in .csv if this is something I've run before.
    # * This is not working properly right now - always detect differences even if running the same thing twice. I'm just writing new files for now

    # %%
    input_param_cols = [k for k in filter_params.keys()] + [
        "beta",
        "Ntot",
        "n_s",
    ]

    # %%
    matching_results_folder = Path(r"C:\Users\ken92\Downloads\Shao-Wen Data\matching_results",
                                f"{geometry}_{geom_spec_str}_spinless_NI_L{L_subregion}")
    matching_results_folder.mkdir(exist_ok = True)
    csv_name = f"ns{np.round(n_s_tar, 4)}.csv"
    csv_path = Path(matching_results_folder, csv_name)

    util_match.update_eos_csv(
        df_new = df_matching_rslt,
        csv_path = csv_path,
        param_cols = input_param_cols,
        bool_overwrite_conflict = True,
    )