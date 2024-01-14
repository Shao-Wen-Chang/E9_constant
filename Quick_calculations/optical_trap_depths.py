from E9_constants import *
from E9_numbers import *
import numpy as np
import matplotlib.pyplot as plt

def um2a(x):
    return x * 1e-6 / a_sw_tri
def a2um(x):
    return x * a_sw_tri / 1e-6

# (532, 1064, ODT):
# (25, 15, 24.05)
# (50, 15, 63.1)
V532nom = 50 # in kHz (i.e. V_SI / hbar / 1e3 / 2pi)
V1064nom = 15
VODTnom = 63.1
attODT = 0.95 # attenuation of ODT as a reference case
halfsize = 250
xrange = np.linspace(-100e-6, 100e-6, halfsize * 2 + 1)
xplotb = xrange * 1e6

V532 = V532nom * Gaussian(xrange, w0_sw)# * np.sin(2 * pi * xrange/lambda_sw)**2
V1064 = - V1064nom * Gaussian(xrange, w0_lw)# * np.sin(2 * pi * xrange/lambda_lw)**2
VODT = - VODTnom * Gaussian(xrange, w0_ODT)
Vlat = V532 + V1064
Vtot = V532 + V1064 + VODT
Vref = V532 + V1064 + VODT * attODT

fig_V = plt.figure(0, figsize = (12, 6))
fig_V.clf()
# fig_V.suptitle('V532 = {} kHz, V1064 = {} kHz, V_ODT = {} kHz, attODT = {}'.format(V532nom, V1064nom, VODTnom, attODT))
ax_V = fig_V.add_subplot(121)
ax_V.plot(xplotb, Vtot, 'g-', label = '$V_{tot}$')
ax_V.plot(xplotb, V532, 'b-', label = '$V_{532}$')
ax_V.plot(xplotb, V1064, 'r-', label = '$V_{1064}$')
ax_V.plot(xplotb, VODT, 'm-', label = '$V_{ODT}$')
ax_V.set_xlabel("x [um]")
ax_V.set_ylabel("V [kHz]")
secax_V = ax_V.secondary_xaxis('top', functions=(um2a, a2um))
secax_V.set_xlabel('x [$a_{lat}$]')
ax_V.legend()

ax_Vfine = fig_V.add_subplot(122)
ax_Vfine.plot(xplotb[int(halfsize * 0.8):int(halfsize * 1.2)], Vtot[int(halfsize * 0.8):int(halfsize * 1.2)] - min(Vtot), 'g-', label = '$V_{tot}$')
ax_Vfine.plot(xplotb[int(halfsize * 0.8):int(halfsize * 1.2)], Vref[int(halfsize * 0.8):int(halfsize * 1.2)] - min(Vref), 'c-', label = '$V_{ODT}$' + ' {} times weaker'.format(attODT))
ax_Vfine.set_xlabel("x [um]")
ax_Vfine.set_ylabel("$(V - V_{min})$ [kHz]")
secax_Vfine = ax_Vfine.secondary_xaxis('top', functions=(um2a, a2um))
secax_Vfine.set_xlabel('x [$a_{lat}$]')
ax_Vfine.legend()