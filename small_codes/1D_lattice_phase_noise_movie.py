"""
Phase-noise driven lattice shift (1D) -> GIF

Two counter-propagating plane waves:
E1 = exp(i(-ω t + k x))
E2 = exp(i(-ω t - k x + p(t)))

Total field: E = E1 + E2
Intensity (ω cancels): I(x,t) = |E|^2 = 2 + 2 cos(2 k x - p(t))

This is a standing wave shifted by x0(t) = p(t)/(2k).

Requires: numpy, matplotlib, pillow (matplotlib uses PillowWriter)
Creates: phase_noise_lattice.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
# Parameters you might tweak
# -----------------------------
gif_name = r"C:\Users\ken92\Documents\Studies\E5\simulation\E9_simulations\small_codes\phase_noise_lattice.gif"

lam = 1.0                    # wavelength units (arbitrary)
k = 2 * np.pi / lam
L = 6 * lam                  # spatial window
Nx = 1200                    # spatial resolution

fps = 20
T_seconds = 2.0              # GIF duration
Nt = int(T_seconds * fps)
dt = 1.0 / fps

# Phase noise model: Ornstein–Uhlenbeck (colored) + optional white jitter
tau = 3000                    # correlation time (s) (bigger -> smoother) (doesn't seem to work)
sigma_phase = 1 * np.sqrt(tau)            # typical phase scale (radians)
white_jitter = 0.          # small extra white phase noise (radians); set 0 to disable

# -----------------------------
# Build p(t): OU process
# -----------------------------
rng = np.random.default_rng(2)

p = np.zeros(Nt)
# OU: dp = -(p/tau) dt + sqrt(2*sigma^2/tau) dW
ou_amp = np.sqrt(2 * sigma_phase**2 / tau)

for n in range(1, Nt):
    dW = np.sqrt(dt) * rng.standard_normal()
    p[n] = p[n - 1] + (-(p[n - 1] / tau) * dt) + ou_amp * dW

if white_jitter > 0:
    p = p + white_jitter * rng.standard_normal(Nt)

# Lattice displacement corresponding to phase
x0 = p / (2 * k)   # units of x

# -----------------------------
# Space grid and intensity
# -----------------------------
x = np.linspace(-L / 2, L / 2, Nx)

def intensity(x, phase):
    # I = 2 + 2 cos(2kx - p(t))
    return 2.0 + 2.0 * np.cos(2 * k * x - phase)

I0 = intensity(x, p[0])

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(8, 5.2))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.28)

ax = fig.add_subplot(gs[0, 0])
axp = fig.add_subplot(gs[1, 0])

(line,) = ax.plot(x / lam, I0, lw=2)
ax.set_xlim(x[0] / lam, x[-1] / lam)
ax.set_ylim(0, 4.1)
ax.set_xlabel("x / λ")
ax.set_ylabel("Intensity  |E|²")
title = ax.set_title("")

# phase trace
t = np.arange(Nt) * dt
axp.plot(t, p, lw=1)
(phase_marker,) = axp.plot([t[0]], [p[0]], marker="o")
axp.set_xlim(t[0], t[-1])
axp.set_xlabel("time (s)")
axp.set_ylabel("p(t) (rad)")

def update(frame):
    Ii = intensity(x, p[frame])
    line.set_ydata(Ii)

    phase_marker.set_data([t[frame]], [p[frame]])

    # Show equivalent displacement in units of λ/2 (lattice period)
    lattice_period = lam / 2
    disp_in_periods = x0[frame] / lattice_period

    title.set_text(
        f"Phase noise shifts standing wave:  p(t)={p[frame]:+.2f} rad   "
        f"x₀(t)=p/(2k)={x0[frame]/lam:+.3f} λ   ({disp_in_periods:+.2f} periods)"
    )
    return line, phase_marker, title

anim = FuncAnimation(fig, update, frames=Nt, interval=1000 / fps, blit=False)

writer = PillowWriter(fps=fps)
anim.save(gif_name, writer=writer)
plt.close(fig)

print(f"Saved GIF to: {gif_name}")
