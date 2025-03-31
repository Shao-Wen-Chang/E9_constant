import numpy as np

class Camera():
    def __init__(self,
                 name: str = "",
                 l_px: float = 0.,
                 CCD_size: tuple[int] = (0, 0),
                 QE: float = 1.,
                 sensor: str = "",
                 framerate: float = 0.,
                 Tmin: float = 0.,
                 company: str = "",
                 model_name: str = "") -> None:
        """Parameters of the camera."""
        self.name = name                # Convenient name of the camera

        # Geometry
        self.l_px = l_px                # [m] size of a single pixel
        self.CCD_size = CCD_size        # (#, #) number of pixels on each side
        self.active_area = np.array([CCD_size[0] * l_px, CCD_size[1] * l_px])

        # (Photo-)Electronic
        self.QE = QE                    # [dimless] Quantum efficiency at ~767 nm
        self.sensor = sensor            # "CCD", "EMCCD", "CMOS"
        self.framerate = framerate      # [Hz] Framerate when using the full imaging area without binning, at highest bit number

        # Others
        self.Tmin = Tmin                # [Celcius] lowest temperature achievable
        self.company = company          # Name of the manufacturing company
        self.model_name = model_name    # Full name of the model

iXon885 = Camera(                       # (x) Andor EMCCD in E3/E4 (Very old model)
    name = "iXon885",
    l_px = 8e-6,
    CCD_size = (1004, 1002),
    QE = 0.5,
    sensor = "EMCCD",
    framerate = 31.4,
    Tmin = -95,
    company = "Andor",
    model_name = "iXon885"
)

iXon888 = Camera(                       # (o) Andor EMCCD
    name = "iXon888",
    l_px = 13e-6,
    CCD_size = (1024, 1024),
    QE = 0.84,
    sensor = "EMCCD",
    framerate = 26,
    Tmin = -95,
    company = "Andor",
    model_name = "iXon888#EXF"
)

iXon897 = Camera(                       # (o) Andor EMCCD
    name = "iXon897",
    l_px = 16e-6,
    CCD_size = (512, 512),
    QE = 0.84,
    sensor = "EMCCD",
    framerate = 56,
    Tmin = -95,
    company = "Andor",
    model_name = "iXon897#EXF"
)

Marana11 = Camera(                      # (x) Andor sCMOS; slightly more thermal noise
    name = "Marana11",
    l_px = 11e-6,
    CCD_size = (2048, 2048),
    QE = 0.7,
    sensor = "CMOS",
    framerate = 24, # 16-bit; 12-bit is "fast" mode, with frame rate 48
    Tmin = -45,
    company = "Andor",
    model_name = "Marana4.2B-11"
)

Marana6 = Camera(                       # (o) Andor sCMOS
    name = "Marana6",
    l_px = 6.5e-6,
    CCD_size = (2048, 2048),
    QE = 0.67,
    sensor = "CMOS",
    framerate = 40, # 16-bit; 12-bit is "low noise" mode, with frame rate 43
    Tmin = -45,
    company = "Andor",
    model_name = "Marana4.2B-6"
)

Stingray = Camera(                      # (o) Allied Vision CCD
    name = "Stingray",
    l_px = 8.3e-6,
    CCD_size = (780, 580),
    QE = 0.1,
    sensor = "CCD",
    framerate = 60,                     # Didn't check which mode
    Tmin = 25,                          # No active cooling
    company = "Allied Vision",
    model_name = "Stingray F-046B"
)

all_cameras = [iXon885, iXon888, iXon897, Marana11, Marana6, Stingray]