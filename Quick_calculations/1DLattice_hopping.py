from E9_constants import *
from E9_numbers import *
import numpy as np
import matplotlib.pyplot as plt

V_lat_z = V0_from_I(6e6, nu_K40_4_2P, f_sw, 4 * I_from_power(0.25, 50e-6), 0, 0)
w_site_z = wsite_from_Vlat(V_lat_z, 2e-6, m_K40)
w_site_lat = wsite_from_Vlat(10e3*hnobar, a_lw_hex, m_K40)
w_site_lat = wsite_from_Vlat(25e3*hnobar, a_sw_tri, m_K40)

Er2kHz = E_R532_Rb87 / hnobar / 1000
thetas = [np.arctan(1/4), np.arctan(1/6), np.arctan(1/9), np.arctan(1/12), np.arctan(1/20)]
Vs = np.linspace(0.2,5)
fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(111)
for theta in thetas:
    ax.plot(Er2kHz * Vs, 1000 * Er2kHz * J_from_Vlat(Vs, theta = theta), label = 'cot(theta) = {}'.format(1/np.tan(theta)))
ax.set_ylim(0, 12)
ax.set_xlabel('$V_{lat}$ (kHz)')
ax.set_ylabel('J (Hz)')
ax.legend()