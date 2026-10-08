import util_match
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import os

#%% Reading data
# Shared values
data_folder = Path(r"C:\Users\ken92\Downloads\Shao-Wen Data")
geometry = "sawtooth" # "sawtooth", "kagome"
chi_tenpy = 64
V_model = 0.3
tp_ST = 1.41421356237
shift_ST = -2.0

# Subregion parameters (User specified)
SRoffset = 2.               # muR - muS
L_subregion = 50            # This is the total number of unit cells, even if bool_muSR is True

# Joint system (Derived from subregion parameters)
L_joint = 100               # This is the total number of unit cells, even if bool_muSR is True

# Shared derived parameters
E_flat_ST = 2. + shift_ST   # energy of the flat band with shift (noninteracting; assuming ymax is None)
E_min = min(E_flat_ST, -4.)
E_max = max(E_flat_ST, 0.)
geom_spec_str = f"V{V_model}_tp{tp_ST}_shift{shift_ST}"
tnpspc = {                  # For checking folder parameter consistency
    "geometry": geometry,
    "L": L_joint,
    "chi": chi_tenpy,
}

#%% Joint system
scan_name_joint = f"ST_SL_I_L{L_joint}_t1_tpsqrt2_shf{int(shift_ST)}_V{V_model}_chi{chi_tenpy}_SRoff{np.round(SRoffset, 1)}"
N_sites_joint = 2 * L_joint

# Check parameter consistency
##### Make below a function #####
folder_params_shared = None
records_EoS = []
scan_folder = Path(data_folder, scan_name_joint)
assert scan_folder.exists()

subfolder_list = [f for f in scan_folder.iterdir() if f.is_dir()]
print(f"found {len(subfolder_list)} folders")

first_betas = None
for i_f, sf_path in enumerate(subfolder_list):
    # Load data and store them as panda data rows
    filepath = Path(sf_path, "measurements.pkl")
    assert filepath.is_file()
    with open(filepath, "rb") as f:
        file_content = pickle.load(f)

    try:
        folder_params = util_match.parse_folder_name(sf_path)
        
        # Check mu consistency between folder name and file
        for s in ['muS', 'muR']: # Assume that mu is specified as muR and muS
            mu = file_content[s]
            if not np.isclose(mu, folder_params[s]):
                raise(ValueError(f"{s}_folder = {folder_params[s]} doesn't agree with mu_file = {mu}"))
            else:
                folder_params.pop(s)
        
        # Check shared folder parameters
        V = folder_params.pop("V")
        shift = folder_params.pop("shift")
        tp = folder_params.pop("tp")
        if folder_params_shared is None:
            folder_params_shared = folder_params.copy()
            for k, v in tnpspc.items():
                if folder_params_shared[k] != v:
                    raise(ValueError(f"mismatch in tenpy parameter {k}: specified {v}, but {folder_params_shared[k]} from folder"))
        else:
            if folder_params != folder_params_shared:
                print(folder_params)
                print(folder_params_shared)
                raise(ValueError("mismatch in folder parameters"))

        # Save data
        for i_b, beta in enumerate(file_content['betas']):
            S2, S2S, S2R, n_avg, n_avgS, n_avgR, energy, energyS, energyR, eps, ov, max_chi, trace, runtime = file_content['data'][i_b, :]
            records_EoS.append({
                "muS": file_content['muS'],
                "muR": file_content['muR'],
                "beta": beta,
                "S2": S2,           # per site
                "S2S": S2S,         # per site
                "S2R": S2R,         # per site
                "n_avg": n_avg,     # per site
                "n_avgS": n_avgS,   # per site
                "n_avgR": n_avgR,   # per site
                "energy": energy,   # per site
                "energyS": energyS, # per site
                "energyR": energyR, # per site
                "eps": eps,
                "ov": ov,
                "max_chi": max_chi,
                "trace": trace,
                "runtime": runtime,
                "file_mtime": os.path.getmtime(filepath),
                "V": V,
                "shift": shift,
                "tp": tp
            })
    except ValueError as e:
        print(filepath)
        raise(e)

    # Check beta length and print warnings if they don't look right
    if first_betas is None:   # assuming that the first subfolder has the right length of data
        first_betas = file_content['betas']
        continue
    else:
        len_good = (len(file_content['betas']) == len(first_betas))      # length is good
        if len_good: # content is also good
            if np.allclose(file_content['betas'], first_betas):
                continue
    print(f"beta doesn't agree for {filepath}")
    print(f"getting {file_content['betas']}")

df_all_data = pd.DataFrame.from_records(records_EoS)
##### Make above a function #####

#%% Filter parameters using Dataframe
##### Make below a function #####
filter_params = {"V": V_model, "shift": shift_ST, "tp": tp_ST}

df_filtered = df_all_data.query(f"(V == {V_model}) & (shift == {shift_ST})")
idx_list = ["muR", "beta"]
df_filtered = df_filtered.set_index(idx_list).sort_index()

mu_vals  = df_filtered.index.get_level_values("muR").unique().to_numpy()
beta_vals = df_filtered.index.get_level_values("beta").unique().to_numpy()
n_grid = df_filtered["n_avg"].unstack("beta").loc[mu_vals, beta_vals].to_numpy()    # particle per site
s2_grid = df_filtered["S2"].unstack("beta").loc[mu_vals, beta_vals].to_numpy()      # Renyi 2-entropy per site
e_grid = df_filtered["energy"].unstack("beta").loc[mu_vals, beta_vals].to_numpy()   # energy per site
eps_grid = df_filtered["eps"].unstack("beta").loc[mu_vals, beta_vals].to_numpy()    # sum of all discarded Schmidt values squared
ov_grid = df_filtered["ov"].unstack("beta").loc[mu_vals, beta_vals].to_numpy()      # lower bound for the overlap
##### Make above a function #####

#%% Plots (joint)
fig_ns, axes_ns = plt.subplots(2, 2, figsize = (8, 7))
for ax, vals, str_label, clim in zip(axes_ns.flatten(),
                    [n_grid, s2_grid, e_grid, 1 - ov_grid],
                    ["n_avg", "S2_avg", "E", "max infidelity"],
                    [None, None, None, (0, 0.01)],):
    img = ax.pcolormesh(beta_vals, mu_vals, vals, clim = clim)
    ax.set_box_aspect(1)
    ax.set_title(str_label)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\mu$")
    # cntr = ax.contour(beta_vals, mu_vals, vals, colors = "white", linestyles = "solid", levels = 8)
    # ax.clabel(cntr, inline = True)
    fig_ns.colorbar(img, ax = ax)
fig_ns.suptitle(f"{geometry} lattice, {geom_spec_str}")
fig_ns.tight_layout()

#%% Repeat the same things for system


#%% Now, compare the difference in EoS between the two
