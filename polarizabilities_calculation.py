from E9_constants import *
from transition_line_data import *
import numpy as np
import matplotlib.pyplot as plt

SI2au = 1/1.64877727436e-41
# wavelengths = np.arange(300, 1000, 0.001) * 1e-9
wavelengths = np.arange(450, 700, 0.001) * 1e-9
alphas = alpha_pol(0, wavelengths, K_gs_lines)
plt.plot(wavelengths, alphas*SI2au)
plt.plot(wavelengths, np.zeros_like(wavelengths))
plt.ylim(-1e4,1e4)
# plt.ylim(-1e4,1e4)