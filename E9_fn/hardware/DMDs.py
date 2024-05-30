class DMD:
    def __init__(self,
                 l_px: float = 0.,
                 DMD_size: tuple[int] = (0, 0),
                 magnification: float = 1.) -> None:
        self.l_px = l_px            # [m] physical size of a single pixel
        self.DMD_size = DMD_size    # (#, #) number of pixels on each side
        self.magnification = magnification      # [dimless] magnification of the imaging system

        self.l_px_atom = l_px * magnification   # [m] size of a pixel in the image plane
        self.active_area_DMD_plane = (DMD_size[0] * l_px, DMD_size[1] * l_px)
        self.active_area_atom_plane = (DMD_size[0] * self.l_px_atom, DMD_size[1] * self.l_px_atom)