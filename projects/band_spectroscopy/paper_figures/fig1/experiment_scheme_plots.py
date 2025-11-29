import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

bool_save_fig = True
# Time axis
t = np.linspace(0, 1, 1000)
t_ramp_up = 0.55
t_hold_end = 0.7
t_shake_end = 0.9
tticks = [0.0, t_ramp_up, t_hold_end, t_shake_end, 1.0]
tticklabels = ['', '', '', '', '']

# Colors
c_int = "#005CEF"
c_pos = "#FF6A00"

#%% Intensity curve (blue, left y-axis)
I = np.zeros_like(t)
I_max = 0.8

# 0 to t_ramp_up: ramp from 0 to t_hold_end (linear for illustration)
mask_1 = (t >= 0.0) & (t < t_ramp_up)
I[mask_1] = I_max * (t[mask_1] / t_ramp_up)

# t_ramp_up to t_hold_end: constant t_hold_end
mask_2 = (t >= t_ramp_up) & (t < t_shake_end)
I[mask_2] = I_max

# t_hold_end to 1.0: back to 0 (already zero by initialization)

#%% Position curve (red, right y-axis)
pos = np.full_like(t, 0.5)

# Oscillating piece: t_hold_end <= t <= t_shake_end
n_periods = 4
t_start = t_hold_end
t_end = t_shake_end
delta_t = t_end - t_start

# Angular frequency omega such that we get 4 periods in [t_hold_end, t_shake_end]
omega = n_periods * 2 * np.pi / delta_t  # this is your "f" in sin(f * T)

osc_mask = (t >= t_start) & (t <= t_end)
pos[osc_mask] = 0.5 + 0.15 * np.sin(omega * (t[osc_mask] - t_start))

#%% Plot
fig, ax1 = plt.subplots(figsize=(5, 4))

# Left axis: intensity
ax1.plot(t, I, color=c_int, lw=2, label='Intensity')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Time')
ax1.set_ylabel('Lattice beam intensity [arb.]', color=c_int)
ax1.set_yticks([])

# Custom x-ticks including t_M between 0.7 and t_shake_end
ax1.set_xticks(tticks)
ax1.tick_params(axis = 'x', labelcolor=c_int, direction = "in")
ax1.set_xticklabels(tticklabels)
text_height = 0.05
ax1.text(
    (t_hold_end + t_shake_end) / 2, text_height,
    r'$t_M$',
    transform=ax1.get_xaxis_transform(),
    ha='center', va='top'
)

# Right axis: position
ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
ax2.set_ylabel('Position [arb.]', color=c_pos)
ax2.tick_params(axis='y', labelcolor=c_pos)
ax2.set_yticks([])

# Plot position in segments to handle dashed part
# 0 to 0.2: dashed at 0.5
mask_pos1 = (t >= 0.0) & (t <= 0.2)
ax2.plot(t[mask_pos1], pos[mask_pos1], color=c_pos, lw=2, ls='--')

# 0.2 to 1: solid
mask_pos2 = (t > 0.2)# & (t < t_hold_end)
ax2.plot(t[mask_pos2], pos[mask_pos2], color=c_pos, lw=2, ls='-')

fig.tight_layout()
plt.show()

fig1apath = Path(r"C:\Users\ken92\Documents\Studies\E5\simulation\E9_simulations\projects\band_spectroscopy\paper_figures\fig1",
                 "fig1b_v1.svg")
if bool_save_fig:
    fig.savefig(fig1apath, format = "svg")