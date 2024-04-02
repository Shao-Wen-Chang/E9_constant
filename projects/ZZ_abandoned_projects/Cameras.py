import numpy as np
# I'm planning to slowly copy things from here to hardware.camera if they
# are useful

#%% Classes
# class Camera():
#     def __init__(self) -> None:
#         # Geometry
#         self.l_px = 0                   # [m] size of a single pixel
#         self.CCD_size = (0, 0)          # (#, #) number of pixels on each side

#         # (Photo-)Electronic
#         self.QE = 0                     # [dimless] Quantum efficiency at ~767 nm

#         # Others
#         self.Tmin = 0                   # [Celcius] lowest temperature achievable

class OpticalSystem():
    def __init__(self) -> None:
        self.M = 0                      # [dimless] Magnification
        self.T = 0                      # [dimless] Transmission at ~767 nm

class SingleExposure():
    def __init__(self) -> None:
        self.exposure_time = 0          # [s] Exposure time
        self.sc_lum = 0                 # [UNDECIDED] Luminosity of the source (photon/s?)
        self.bg_lum = 0                 # [UNDECIDED] Luminosity of the background (photon/s?)

class SingleImageProperties():
    def __init__(self) -> None:
        self.signal = 0                 # [UNDECIDED] Strength of signal (photoelectron count?)
        self.noise = 0                  # [UNDECIDED] Strength of noise (photoelectron count?)
        self.FoV = 0                    # [m] Field of view


#%% Functions
def acquire_image(cam, opt_sys, exposure):
    img = SingleImageProperties()
    return img

#%% main
frac_collected = (1 - np.cos(np.pi/6))/2
