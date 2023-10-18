from E9_constants import *
from E9_numbers import *
import numpy as np
import matplotlib.pyplot as plt

thetas = [np.arctan(1/4), np.arctan(1/6), np.arctan(1/9), np.arctan(1/12), np.arctan(1/20)]
V_lat = np.arange(10e3, 100e3, 1e3)
V_lat2 = np.arange(20e3, 500e3, 2e3)
fx_532tri = wsite_from_Vlat(hnobar*V_lat*(8/9), a_sw_tri, m_K40)/2/np.pi
fx_1064hex = wsite_from_Vlat(hnobar*V_lat*(1/9), a_lw_hex, m_K40)/2/np.pi
fz_1064vert = wsite_from_Vlat(hnobar*V_lat, a_vert, m_K40)/2/np.pi

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax.plot(V_lat / 1000, fx_532tri / 1000, label = '532 triangular')
ax.plot(V_lat / 1000, fx_1064hex / 1000, label = '1064 honeycomb')
ax.plot(V_lat / 1000, fz_1064vert / 1000, label = '1064 vertical')
for theta in thetas:
    fz_532vert = wsite_from_Vlat(hnobar*V_lat2, 532e-9/2/np.sin(theta), m_K40)/2/np.pi
    ax2.plot(V_lat2 / 1000, fz_532vert / 1000, label = '532 vertical, cot(theta) = {}'.format(1/np.tan(theta)))
ax.set_xlabel('$V_{lat}$ (kHz)')
ax.set_ylabel('$f_{site}$ (kHz)')
ax.legend()
ax2.legend()