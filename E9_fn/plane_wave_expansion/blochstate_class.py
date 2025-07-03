import numpy as np
from scipy.linalg import eigh, expm
import copy
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import seaborn as sns

sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
import E9_fn.E9_constants as E9c
from E9_fn import util

#%% blochstate (the class)
class blochstate(np.ndarray):
    """A numpy array-like object that stores information about the bloch state.
    
    blochstate[m, n] is the (possibly complex) coefficient of the (m * g1 + n * g2) component.
    While the bloch state is a 2D-array, for calculation purposes I often need to convert it into an
    1D-array. The state also carries information like what are the parameters used in calculating them.
    The magic methods are mostly copied from random internet guy and I don't know how they work.
    To add more attributes while making sure that the states pickle correctly:
        1. Add obj.(attr) = (input argument or fixed value) in __new__
        2. in __array_finalize__, add self.(attr) = getattr(obj, '(attr name string)', default value)"""
    def __new__(cls,
                input_array,
                q: tuple = (0, 0),
                center: tuple = (0, 0),
                N: int = 0,
                E: float = 0.,
                error: float = 0,
                info: str = "",
                param: dict = {}):
        """Read more to figure out what I am doing here! It's all from
        https://numpy.org/devdocs/user/basics.subclassing.html#module-numpy.doc.subclassing
        https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
        """
        # I *guess* __init__ is not needed ...?
        obj = np.asarray(input_array).view(cls)
        obj.q = q           # this is normalized by Kp, so (0, 1) is exactly on kappa point
        obj.center = copy.deepcopy(center) # can be useful if the state is known to be asymmetric
        obj.N = N           # band number (physical meaning is less clear when swallow tails appear)
        obj.E = E           #(chemical) energy
        obj.ksize = int(np.sqrt(len(input_array)))  # size of k-space
        obj.num = int((obj.ksize - 1) / 2)          # (also given as (-num, num))
        obj.error = error   # the self-consistent error of the state
        obj.vx = []         # vx log during calculation; seldom used
        obj.vy = []         # vy log during calculation; seldom used
        obj.info = copy.deepcopy(info)     # misc info, such as methods used for calculation
        obj.param = copy.deepcopy(param)   # parameters used in simulation
        # A list of parameters to be included: {'V532', 'V1064', 'n0nom'(, 'V532pol', 'V1064pol', 'Vdis')}
        # By 'pol' I mean Vin or Vout; we probably don't need to bookkeep this
        # 'Vdis': relative displacement between the two lattices, will probably rewrite code anyways if not 0
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.q = getattr(obj, 'q', (0,0))
        self.center = getattr(obj, 'center', (0,0))
        self.N = getattr(obj, 'N', 0)
        self.E = getattr(obj, 'E', 0.)
        self.ksize = getattr(obj, 'ksize', 1)
        self.num = getattr(obj, 'num', 1)
        self.error = getattr(obj, 'error', 0)
        self.vx = getattr(obj, 'vx', [])
        self.vy = getattr(obj, 'vy', [])
        self.info = getattr(obj, 'info', "created via view or from template")
        self.conv = getattr(obj, 'conv', 0)
        self.param = getattr(obj, 'param', {})
    
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(blochstate, self).__reduce__()
        # Create our own tuple to pass to __setstate__, but append the __dict__ rather than individual members.
        new_state = pickled_state[2] + (self.__dict__,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)
    
    def __setstate__(self, state):
        self.__dict__.update(state[-1])  # Update the internal dict from state
        # Call the parent's __setstate__ with the other tuple elements.
        super(blochstate, self).__setstate__(state[0:-1])
        
    # Index manipulation methods
    def mn2index(self, m, n, quiet = False):
        """Given some m, n (labels for plane wave components), return the corresponding index for the (1D-)Blochstate."""
        goodm = m in range(-self.num + self.center[0], self.num + self.center[0] + 1)
        goodn = n in range(-self.num + self.center[1], self.num + self.center[1] + 1)
        if not (goodm and goodn):
            if not quiet: print("({}, {}) is out of range".format(m, n))
            return
        return m * self.ksize + n
    
    def mnasindex(self, m, n):
        """A shorthand for state[state.mn2index(m, n)]"""
        return self[self.mn2index(m, n)]
    
    def index2mn(self, index, quiet = False): # need to test, likely shifted somehow
        # just look up modulo...
        dg1 = self.center[0]
        dg2 = self.center[1]
        if index > self.ksize * (self.num + dg1) + self.num + dg2: index -= self.ksize**2
        offset = 2 * self.num * (self.num - dg1 + 1) - (dg1 + dg2)
        m = (index + offset) // self.ksize + dg1 - self.num
        n = (index + offset) % self.ksize + dg2 - self.num
        goodm = m in range(-self.num + dg1, self.num + dg1 + 1)
        goodn = n in range(-self.num + dg2, self.num + dg2 + 1)
        if not (goodm and goodn):
            if not quiet: print("{} is out of range; ({}, {})".format(index, m, n))
            return
        return (m, n)
    
    def rotate(self, direction = 1, parity = 1):
        """"rotate" the state cw by (direction * 60) deg.

        Set parity = -1 to get the mirror of the (rotated) state.
        (should be convenient for e.g. getting q = -1 from q = 1)
        """
        def map2newmn(m, n, direction, parity):
            if direction == 0:
                return (m * parity, n * parity)
            elif direction == 1:
                return (n * parity, (n - m) * parity)
            elif direction == -1:
                return ((m - n) * parity, m * parity)
        
        new_bloch = copy.deepcopy(self)
        for i in range(len(new_bloch)): new_bloch[i] = 0
        dg1 = self.center[0]
        dg2 = self.center[1]
        for m in range(-self.num + dg1, self.num + dg1 + 1):
            for n in range(-self.num + dg2, self.num + dg2 + 1):
                mm, nn = map2newmn(m, n, direction, parity)
                i = self.mn2index(m, n, quiet = True)
                ii = self.mn2index(mm, nn, quiet = True)
                if ii != None:
                    new_bloch[ii] = self[i]
                    #print("({}, {}): ii = {}, value = {}".format(mm, nn, ii, self[i]))
        return new_bloch
    
    def _crop(self, num2): # unfinished
        """Crops the state to the specified size (and change relevant attributes)."""
        if self.num <= num2:
            print("Original state is too small to crop. State is unchanged")
        else:
            pass # return cropped state, but leave state unchanged
    
    def _pad(self, num2, quiet = False): # unfinished - not working yet???
        # return a state padded to num2, filling additional values with zeros (implementation not smart but works)
        if num2 <= self.num:
            if not quiet: print("original state is larger or equal")
        else:
            # try nonlocal
            new_ksize = 2 * num2 + 1
            new_array = np.zeros((2 * num2 + 1)**2)
            new_bloch = copy.deepcopy(self)
            for m in range(-self.num, self.num + 1):
                for n in range(-self.num, self.num + 1):
                    new_array[m * new_ksize + n] = self[m * self.ksize + n]
            new_bloch.ksize = new_ksize
            new_bloch.num = num2
            return new_array
    
    # Methods for retrieving physical values
    def population(self):
        """finds momentum space distribution (basically just returns a blochstate of <phi|phi>)"""
        return abs(self)**2
        
    def findvx(self):
        """finds the group velocity in x direction (normalized s.t. kp = 1)
        
        Should combine with findvy when I care enough.
        """
        p = self.population()
        dg1 = p.center[0]
        dg2 = p.center[1]
        g1x = E9c.g1[0]
        g2x = E9c.g2[0]
        # a "velocity mask", independent of population
        mg = np.mgrid[(-p.num + dg1):(p.num + dg1 + 1), (-p.num + dg2):(p.num + dg2 + 1)]
        vxmask = mg[0] * g1x + mg[1] * g2x
        # First move the center element (given by self.center) of the population to the middle of the state array, then reshape it
        # to a square. Finally multiply element-wise by the velocity mask, and sum all entries
        # (see my onenote for a schematic derivation...)
        return self.q[0] + (float((np.roll(p, 2 * self.num * (self.num - dg1 + 1) - (dg1 + dg2)).reshape(self.ksize, self.ksize) * vxmask).sum()) / E9c.k_sw)
        
    def findvy(self):
        """finds the group velocity in y direction (normalized s.t. kp = 1)"""
        p = self.population()
        dg1 = p.center[0]
        dg2 = p.center[1]
        g1y = E9c.g1[1]
        g2y = E9c.g2[1]
        mg = np.mgrid[(-p.num + dg1):(p.num + dg1 + 1), (-p.num + dg2):(p.num + dg2 + 1)]
        vymask = mg[0] * g1y + mg[1] * g2y
        return self.q[1] + (float((np.roll(p, 2 * self.num * (self.num - dg1 + 1) - (dg1 + dg2)).reshape(self.ksize, self.ksize) * vymask).sum()) / E9c.k_sw)
    
    def findj(self, x, y):
        """Find the current density.
        
        See PHYSICAL REVIEW A 86, 063636 (2012).
        xx and yy are the x and y axis used to generate grids in real space, where the currents are calculated.
        For [len(xx), len(yy)] = [X, Y], the output jout is a (2, X, Y)-dimensional ndarray, where jout[0,g,h] and
        jout[1,g,h] are the x- and y- component of the current at point (xx[g], yy[h])."""
        num, size, dg1, dg2 = self.num, self.ksize, self.center[0], self.center[1]
        xx, yy = np.meshgrid(x, y)
        jout = np.zeros((2, len(x), len(y)), dtype = np.complex128)
        
        for m in range(-num + dg1, num + dg1 + 1):
            for n in range(-num + dg2, num + dg2 + 1):
                for mm in range(-num + dg1, num + dg1 + 1):
                    for nn in range(-num + dg2, num + dg2 + 1):
                        K = m * E9c.G1 + n * E9c.G2 + self.q
                        KK = (mm - m) * E9c.G1 + (nn - n) * E9c.G2 + self.q
                        Mat = np.conj(self[mm * size + nn]) * self[m * size + n] * np.cos(KK[0] * xx + KK[1] * yy)
                        jout[0,:,:] += K[0] * Mat
                        jout[1,:,:] += K[1] * Mat
        if not util.IsReal(jout): raise Exception("imaginary current")
        return np.real(jout)
    
    # Plot related methods
    def realplot(self, sample_size = 100):
        """finds the real space distribution (return: (sample_size, sample_size) 2d array)"""
        RealPlot(self, sample_size = sample_size)

    # Other convenience methods
    def totext(self):
        """Returns a human readable string that focuses on blochstate labels, i.e. |q, n>.
        
        I decided to not modify __str__ because I sometimes still print the whole state out to check its elements.
        """
        qx, qy, N = self.q[0], self.q[1], self.N
        return "|[{:.4},{:.4}], {}>".format(qx, qy, N)

#%% Functions for manipulating (list of) blochstate objects
# I didn't make good judgement on what should be standalone functions and what
# should be class methods
def ShiftCenter(psi, new_center):
    """move the center of the state; indices originally outside of consideration are replaced by 0"""
    new_psi = copy.deepcopy(psi)
    dg1 = psi.center[0]
    dg2 = psi.center[1]
    dg1new = new_center[0]
    dg2new = new_center[1]

    for m in range(-psi.num + dg1new, psi.num + dg1new + 1):
        for n in range(-psi.num + dg2new, psi.num + dg2new + 1):
            if m in range(-psi.num + dg1, psi.num + dg1 + 1) and n in range(-psi.num + dg2, psi.num + dg2 + 1):
                new_psi[m * psi.ksize + n] = psi[m * psi.ksize + n]
            else:
                new_psi[m * psi.ksize + n] = 0
    new_psi.center = new_center
    return new_psi

def FindInStateList(slist, q, N):
    """Given a list of blochstate, pick out the state that has the wanted q and N.
    
    This function needs to be updated to be useful."""
    if type(q) != np.ndarray: q = np.array([0, q])
    for s in slist:
        if type(s) != blochstate:
            continue
        elif np.allclose(s.q, q) and s.N == N:
            #print("found state with q = {0}, N = {1}; E = {2}".format(s.q, s.N, s.E))
            return s
    print("state with q = {0} and N = {1} not found".format(q, N))

def SortStateListBy(slist, sortstr, order = 1):
    """Sort the blochstates in a list by some acceptable parameters.
    
    order = 1 sorts the list in increasing order; -1, decreasing. States with the same parameters are not sorted.
    This should be defined in a way that's more or less general, so that even if I add some other parameters in param
    or attribute I can still sort by the new parameters."""
    pass

def SaveVariables(filepath, *args, saveall = False):
    """Save all the variables in a dictionary.
    
    This is still ill defined because I don't know how modules work.
    The inputs are specified as a string of their names. If saveall = True, then globals() is saved.
    This should replace SaveStateList as the preferred way of saving items."""
    save_dic_temp = {}
    gl = vars(sys.modules['__main__'])
    # print(gl.keys())
    if saveall:
        save_dic_temp = gl
    else:
        for varstr in args:
            save_dic_temp[varstr] = copy.deepcopy(gl[varstr])
    save_dic = copy.deepcopy(save_dic_temp)
    with open(filepath, "wb") as f:
        pickle.dump(save_dic, f)

def LoadVariables(filepath):
    """Load the variables saved by SaveVariables.
    
    This is really just pickle.load. This should replace LoadStateList for readability."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

def SaveStateList(filepath, slist):
    # Saves the whole list; in principle the elements don't have to be blochstates.
    # e.g. One can also store a ndarray with this function
    
    with open(filepath, "wb") as f:
        pickle.dump(slist, f)

def LoadStateList(filepath):
    # need to extend a lot if I use State class
    with open(filepath, "rb") as f:
        return pickle.load(f)

def SaveStateListAsCsv(filepath, slist, eol_str = '\n'):
    """Save list as a csv file; each line is a separate array"""
    if filepath[-4:] != '.csv': filepath += '.csv'
    with open(filepath, "a") as file:
        for s in slist:
            s.tofile(file, sep = ',', format = '%e')
            file.write(eol_str)

#%% More "physics related" functions
def FindDensity(psi, xx = np.arange(-1.5, 1.5, 0.025), yy = np.arange(-1.5, 1.5, 0.025)):
    """Returns the real space density distribution and wave fn phases of the given blochstate.
    
    Density is normalized such that the maximum element is 1, while there is no fixed convention for phase."""
    mg = np.meshgrid(xx, yy)
    def SumPlaneWavesForOnePsi(psi_in):
        """Generates the real wave fn by summing plane waves (for one blochstate)."""
        psi_in_r = np.zeros((len(xx), len(yy)), dtype = np.cdouble)
        for index in range(len(psi_in)):
            m, n = psi_in.index2mn(index)
            q = psi_in.q
            k_mn = m * G1 + n * G2 + q
            psi_in_r += psi_in[index] * np.exp(1j * (k_mn[0] * mg[0] + k_mn[1] * mg[1] ))
        return psi_in_r
    
    if type(psi) == blochstate:
        psi_r = SumPlaneWavesForOnePsi(psi)
    elif type(psi) == LinSupState:
        psi_r = np.zeros((len(xx), len(yy)), dtype = np.cdouble)
        for s, r in zip(psi.states, psi.weights):
            psi_r += r * SumPlaneWavesForOnePsi(s) / psi.norm_factor
    density = abs(psi_r)**2
    density = density / density.max()
    phase = np.angle(psi_r)
    return density, phase

def FindDoS(e_values, samp = 1, cellsize = 1):
    '''Computes the density of state.
    
    This function requires one to compute the band structure over the entire Brillouin zone. Computation can be simplified
    if there exists some symmetries, e.g. for an undistorted Kagome lattice one only needs to compute 1/12 of the Brillouin
    Zone. It is up to the user to define the correct momentum space of interest.
    Here, the DoS at energy E is simply the number of values in e_values within (E, E + dE). One can get the correct dimen-
    sional value by providing a value for cellsize.
    Args:
        e_values: a (num, N)-dim ndarray, where e_values[nq, i] is the energy of the state with quasimomentum nq in the i-th
                  band. Since this array is flattened throughout the function, the actual shape of the array is immaterial.
                  The important thing is that the quasimomentum points should be sampled uniformly in each band, with the
                  same sampling rate (which is guaranteed in my code).
        samp: how fine is E_DoS interval defined. Number of intervals is (default * samp).
        cellsize: the size of phase space cell (which depends on the sampling rate of momentum space). Also multiply by a
                  symmetry factor if only part of the BZ were used in computation.
    Returns:
        E_DoS: nE-dim array of energies at which the DoS is calculated.
        DoS: nE-dim array; DoS calculated at E_DoS. DoS[i] is the number of states with energy within [E_DoS[i], E_DoS[i + 1]).
        DoS_int: nE-dim array; integrated DoS. In other words, DoS_int[i] is the number of states with energy within
                 [E_DoS[0], E_DoS[i + 1]).'''
    E_DoS = np.linspace(e_values.min(), e_values.max(), int(e_values.size / 10 * samp))
    DoS, DoS_int = np.zeros_like(E_DoS), np.zeros_like(E_DoS)
    for i, E_higher in enumerate(E_DoS):
        DoS_int[i] = (e_values <= E_higher).astype(int).sum() * cellsize
    DoS = np.hstack((DoS_int[1:], np.array(DoS_int[-1]))) - DoS_int
    return E_DoS, DoS, DoS_int

#%% Plot related stuff
# All these V-related functions should be updated
def Vin(x_, y_, V, dx = 0, dy = 0, lamb = 1):
    """Plot in-plane polarization potential (lamb stands for lambda)"""
    x, y = x_ - dx, y_ - dy
    xx, yy = np.meshgrid(x, y)
    return (2/9) * V * (3 - np.cos((E9c.g1[0] * xx + E9c.g1[1] * yy) / lamb)
                          - np.cos((E9c.g2[0] * xx + E9c.g2[1] * yy) / lamb)
                          - np.cos((E9c.g3[0] * xx + E9c.g3[1] * yy) / lamb))

def Vout(x_, y_, V, dx = 0, dy = 0, lamb = 1):
    """Plot out of plane polarization potential"""
    x, y = x_ - dx, y_ - dy
    xx, yy = np.meshgrid(x, y)
    return (1/9) * V * (3 + 2 * np.cos((E9c.g1[0] * xx + E9c.g1[1] * yy) / lamb)
                          + 2 * np.cos((E9c.g2[0] * xx + E9c.g2[1] * yy) / lamb)
                          + 2 * np.cos((E9c.g3[0] * xx + E9c.g3[1] * yy) / lamb))

def Vsuper(V532, Pol532, V1064, Pol1064, x, y, dx, dy):
    """Plot combined potential (Pol stands for polarization)"""
    return Pol532(x, y, V532, 0, 0, 1) - Pol1064(x, y, V1064, dx, dy, 2)

def VSubPlot(ax, V, Exp_lib):
    V532nom, V1064nom = Exp_lib['V532nom'], Exp_lib['V1064nom']
    ax.set_title(r'$V532 =${0}, $V1064 =${1}'.format(V532nom, V1064nom))
    return ax.contourf(V) # returns the mappable generated by contourf for colorbar etc.

def VPlot(V):
    fig = plt.figure(5, figsize=(7,7))
    fig.clf()
    ax = fig.add_subplot()
    VSubPlot(ax, V)

def RealSubPlot(ax, psi, x, y, lev = 8, plotrange = False):
    """Plot the real space distribution and phases of the given blochstate.
    
    ax should be a list with two axeses: ax = [ax1, ax2]
    xx, yy = np.arange(-1, 1, 0.02), np.arange(-1, 1, 0.02) shows one full hexagon of the 1064 honeycomb lattice.
    The default np.arange(-1.5, 1.5, 0.025) shows roughly 7 plaquettes.
    """
    ax1, ax2 = ax
    if type(psi) == blochstate: ax1.set_title("(q,N) = ([{:.4},{:.4}],{})\nE = {:.4}".format(psi.q[0], psi.q[1], psi.N, psi.E))
    
    density, phase = FindDensity(psi, x, y)
    if plotrange: # Change color level scaling
        rmin, rmax = plotrange
        cnt = ax1.contourf(x, y, density, lev, vmin = rmin, vmax = rmax)
    else:
        cnt = ax1.contourf(x, y, density, lev)
    cms = ax2.pcolormesh(phase, cmap = ListedColormap(sns.color_palette('husl', 256)),
                         vmin = -np.pi, vmax = np.pi, zorder = 1)
    ax2.imshow(np.ones_like(phase), cmap = ListedColormap(['#FFFFFF']),
               alpha = util.LogisticFn(1 - density, x0 = 0.6, k = 8),
               origin = 'lower', zorder = 2)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax2.tick_params(bottom = False, left = False, labelbottom = False, labelleft = False)
    return [cnt, cms]

def jSubPlot(ax, psi, xin, yin, sampling_rate):
    """Plot the real space current density for a blochstate psi."""
    xj, yj = xin[::sampling_rate], yin[::sampling_rate]
    j_psi = psi.findj(xj, yj)
    j_norm = 1 / (np.sqrt(j_psi[0]**2 + j_psi[1]**2).max() * 8)
    j_psi = j_psi * j_norm
    for m, x in enumerate(xj):
        for n, y in enumerate(yj):
            jx, jy = j_psi[0][m][n], j_psi[1][m][n]
            ax.arrow(x, y, jx, jy, edgecolor = 'blue', facecolor = 'blue', width = 0.007, head_width = 0.03
                     , head_length = 0.02, overhang = 0, length_includes_head = True)
    ax.set_aspect('equal')
    ax.set_title('(x {:.4})'.format(j_norm))
    ax.set_xlim(xin[0], xin[-1])
    ax.set_ylim(yin[0], yin[-1])

def RealPlot(psilist, x = np.arange(-1.5, 1.5, 0.025), y = np.arange(-1.5, 1.5, 0.025), lev = 8, plotrange = False, plotj = False):
    """Plot the real space distribution of blochstates in psilist."""
    if type(psilist) != list: psilist = [psilist]
    ncols = len(psilist) + 1
    fig = plt.figure(4, figsize=(5,5))
    fig.clf()
    for i, psi in enumerate(psilist):
        if plotj:
            num_row = 3
            sampling_rate = 5 # only plot an arrow for every sampling_rate elements
            ax1, ax2, ax3 = fig.add_subplot(3, ncols, i+1), fig.add_subplot(3, ncols, i+ncols+1), fig.add_subplot(3, ncols, i+2*ncols+1)
            jSubPlot(ax3, psi, x, y, sampling_rate)
        else:
            num_row = 2
            ax1, ax2 = fig.add_subplot(2, ncols, i+1), fig.add_subplot(2, ncols, i+ncols+1)
        cnt, cms = RealSubPlot([ax1, ax2], psi, x, y, lev, plotrange)
    fig.colorbar(cnt, ax = fig.add_subplot(num_row, ncols, ncols))
    fig.colorbar(cms, ax = fig.add_subplot(num_row, ncols, 2 * ncols))
    return fig
        
def ToFSubplot(ax, vec, center = (0, 0), maxmn = 3):
    """"Plot ToF images in a given axes.
    
    (versions after 211107) I use two different color tables: if all entries are real, I use black and blue for
    different parities, and red to indicate an actually small element that's somehow displayed. If there are any
    imaginary numbers, then I use some cyclic color palette.
    """
    size = int(np.sqrt(len(vec)))
    num = int((size - 1) / 2)
    maxmn = min(maxmn, num)
    if type(vec) == blochstate: center = vec.center
    vec_is_complex = bool(np.any(abs(np.imag(vec))) > 1e-5)
    dg1 = center[0]
    dg2 = center[1]
    plotsize = 2 * maxmn + 1
    x = np.zeros(plotsize**2)
    y = np.zeros(plotsize**2)
    weight = np.zeros(plotsize**2) # values for dot sizes
    colors = np.zeros(plotsize**2)   # values for colormaps
    i = 0
    # color array is defined as the polar angle of entries
    if not vec_is_complex: # positive entries has angle = 0, and negative entries pi. Values close to 0 are assined pi/2
        cmp = ListedColormap(['black', 'red', 'blue'])
    else:
        # common choices: cm.get_cmap('twilight'); cm.get_cmap('hsv'); ListedColormap(sns.color_palette('husl', 256))
        cmp = ListedColormap(sns.color_palette('husl', 256))
    
    for m in range(-maxmn + dg1, maxmn + dg1 + 1):
        for n in range(-maxmn + dg2, maxmn + dg2 + 1):
            index = m * size + n
            (x[i], y[i]) = E9c.G1 * m + E9c.G2 * n
            weight[i] = abs(vec[index]) ** 2 * 800 # remember to square the wavefn (basically we measure |<psi|e^ikx>|^2)
            if np.allclose(abs(vec[index]), 0):
                colors[i] = np.pi/2
            else:
                colors[i] = np.angle(vec[index])
            i += 1
    
    if type(vec) == blochstate:
        ax.set_title("(q,N) = ([{:.4},{:.4}],{}), E = {:.4}".format(vec.q[0], vec.q[1], vec.N, vec.E))
        q = vec.q * E9c.k_lw
        ax.scatter(q[0], q[1], s = 10, c = "red", marker = "x", zorder = 3)
    sca = ax.scatter(x, y, s = weight, c = colors, cmap = cmp, zorder = 1)
    ax.set_aspect('equal')
    return sca

def ToFPlot(psilist, nrows = 2, maxmn = None):
    """Get ToF plot for all states in psilist, plotted in that order.
    
    Note that q is not considered here, so the center blob actually has a momentum hbar*q.
    The last axes is used to place colorbar (of the last state)."""
    if type(psilist) != list: psilist = [psilist]
    nomaxmn = (maxmn == None)
    ncols = (2 + len(psilist)) // nrows
    fig = plt.figure(2, figsize = (ncols*6, nrows*4))
    fig.clf()
    for i, psi in enumerate(psilist):
        ax = fig.add_subplot(nrows, ncols, i+1)
        if type(psi) == blochstate and nomaxmn:
            maxmn = psi.num
        elif nomaxmn:
            maxmn = 3
        sca = ToFSubplot(ax, psi, maxmn = maxmn)
    fig.colorbar(sca, ax = fig.add_subplot(nrows, ncols, nrows * ncols))

# check if I want to keep this
def PlotEnergyFunctional(ax, datalist, marker = '-', label = '', color = 'r', index_list = None):
    """(Assuming the size and center of all states in the list are the same,) plot the energy functional
    on a given axes (index_list can be given to speed up plotting)."""
    if index_list == None:
        # set index_list = [] to plot non-interacting solutions
        # note that using a different U0 from that used to find interacting solution is meaningless
        psi = datalist[0]
        index_list = FindInteractionIndices(psi.num, center = psi.center)
    energies = [GPEFunctional(psi, index_list) for psi in datalist]
    xq = [psi.q[1] for psi in datalist]
    ax.plot(xq, energies, marker, label = label, color = color)

# check if I want to keep this
def PlotBZSubplot(ax_BZ, qset = '', BZcolor = E9c.BZcolor_PRL):
    """Plot the Brillouin zone of lattice(, and the quasimomentum path / area if given).
    
    By default the quasimomentum plotted is normalized by K.
    Args:
        qset: there are two different possible kinds of inputs
                  i) string: if q-points are defined along some path. e.g. 'Kp/K, Gp/K, Mp/K, Kp/K'
                     In this case, a series of arrows indicate the quasimomentum path.
                 ii) a tuple of (string, points) where points is an array obtained from FindqArea(eval(qvert))
                     Here the area of interest is shaded, and each point is marked in the BZ.
              This input assumes that the evaluated elements are normalized by K."""
    # Determine what input type was given
    if qset == '':
        qset_type = 0
    elif type(qset) == str:
        qset_type = 1
        qstr = qset
        q_verts = eval(qstr)
    elif type(qset) == tuple:
        qset_type = 2
        qstr, q_pts = qset
        q_verts = eval(qstr)
    
    # Define Path objects for BZ
    arrow_color = '#FF5500'#'#FAB16C'
    xx, yy = np.meshgrid(np.arange(-4, 4), np.arange(-4, 4))
    path1 = util.GetClosedPolygon(E9c.BZ1_vertices)
    path2 = util.GetClosedPolygon(E9c.BZ2_vertices)
    path3 = util.GetClosedPolygon(E9c.BZ3_vertices)
    path4 = util.GetClosedPolygon(E9c.BZ4_vertices)
    patch1 = patches.PathPatch(path1, facecolor = BZcolor[1], lw=2, alpha = 1)
    patch2 = patches.PathPatch(path2, facecolor = BZcolor[2], lw=2, alpha = 1)
    patch3 = patches.PathPatch(path3, facecolor = BZcolor[3], lw=2, alpha = 1)
    patch4 = patches.PathPatch(path4, facecolor = BZcolor[4], lw=2, alpha = 1)
    ax_BZ.add_patch(patch4)
    ax_BZ.add_patch(patch3)
    ax_BZ.add_patch(patch2)
    ax_BZ.add_patch(patch1)
    
    # Add some points in the reciprocal lattice
    for i in xx:
        for j in yy:
            x = i * E9c.G1G[0] + j * E9c.G2G[0]
            y = i * E9c.G1G[1] + j * E9c.G2G[1]
            ax_BZ.plot(x, y, 'ok')
            if qset_type == 1: # Mark equivalent quasimomenta for final quasimomentum
                x = i * E9c.G1G[0] + j * E9c.G2G[0] + q_verts[-1][0]
                y = i * E9c.G1G[1] + j * E9c.G2G[1] + q_verts[-1][1]
                ax_BZ.plot(x, y, 'or', markersize = 3)
    
    # Plot the quasimomentum path / area
    if qset_type == 1:
        for i in range(len(q_verts) - 1):
            x, y = q_verts[i]
            dx, dy = (q_verts[i + 1] - q_verts[i])
            ax_BZ.arrow(x, y, dx, dy, edgecolor = arrow_color, facecolor = arrow_color, width = 0.02
                        , head_width = 0.15 , head_length = 0.25, overhang = 0.5, length_includes_head = True)
    elif qset_type == 2:
        polypath = util.GetClosedPolygon(q_verts)
        patchq = patches.PathPatch(polypath, facecolor = BZcolor[0], lw = 1, alpha = 0.4)
        ax_BZ.add_patch(patchq)
        for pt in q_pts:
            ax_BZ.plot(pt[0], pt[1], '.r', markersize = 1.5)
    
    # Small stuff
    ax_BZ.arrow(-2 * (E9c.G1G[0] + E9c.G2G[0]), (-2 * E9c.G1G[1] - 3 * E9c.G2G[1])
                , E9c.kB12[0] / E9c.k_lw, E9c.kB12[1] / E9c.k_lw, edgecolor = arrow_color
                , facecolor = arrow_color, width = 0.02, head_width = 0.15 , head_length = 0.25
                , overhang = 0.5 , length_includes_head = True)
    ax_BZ.arrow(-2 * (E9c.G1G[0] + E9c.G2G[0]), (-2 * E9c.G1G[1] - 3 * E9c.G2G[1])
                , E9c.kB23[0] / E9c.k_lw, E9c.kB23[1] / E9c.k_lw, edgecolor = arrow_color
                , facecolor = arrow_color, width = 0.02, head_width = 0.15 , head_length = 0.25
                , overhang = 0.5 , length_includes_head = True)
    ax_BZ.set_xlim(-4, 4)
    ax_BZ.set_ylim(-3, 3)
    ax_BZ.set_aspect('equal')
    plt.pause(0.01)
    return ax_BZ

def PlotBZ(qset = '', BZcolor = E9c.BZcolor_PRL, fignum = 100):
    """Convienent function for plotting BZ."""
    fig = plt.figure(fignum)
    fig.clf()
    ax_BZ = fig.add_subplot(111)
    PlotBZSubplot(ax_BZ, qset = qset, BZcolor = BZcolor)
    if type(qset) == str: # see PlotBZSubplot
        qstr = qset
    else:
        qstr = qset[0]
    fig.suptitle('qset: ' + qstr)
    return ax_BZ

def FindQAxis(num, qset):
    """Return the q-axis we use to plot e.g. dispersion in the usual convention."""
    q_last = np.linalg.norm(qset[0])
    qaxis = np.array([q_last])
    for i in range(len(qset[1:])):
        q_abs = np.linalg.norm(qset[i+1] - qset[i])
        qaxis = np.concatenate((qaxis, np.linspace(q_last, q_last + q_abs, num[i])[1:])) # cut the first point according to my convention
        q_last += q_abs
    return qaxis

def FindqSet(num, p1, p2):
    """Gives intermediate points between two points."""
    return np.array([p1 * (1-i) + p2 *i for i in np.linspace(0, 1, num = num)])

def FindqSets(num, points):
    """Gives intermediate points for a tuple of points."""
    def GiveNum(i):
        if type(num) == int: return num
        else: return num[i]
    qlist = FindqSet(GiveNum(0), points[0], points[1])
    for i in range(1, len(points) - 1):
        qlist = np.vstack((qlist, FindqSet(GiveNum(i), points[i], points[i+1])[1:]))
    return qlist

def FindqArea(qvert, dqx = 0.015, dqy = 0.015):
    '''Gives points within an area defined by a set of vertices.
    
    Args:
        qvert: a tuple of points like np.array([qx, qy]) that defines a polygon. Shapes with as many horizontal / vertical
               lines are preferred.
        dqx, dqy: the distance between neighboring points.'''
    # Define the polygon and find the smallest covering rectangle
    polypath = util.GetClosedPolygon(qvert)
    vxs, vys = [pt[0] for pt in qvert], [pt[1] for pt in qvert]
    xmin, xmax, ymin, ymax = min(vxs), max(vxs), min(vys), max(vys)
    x, y = np.arange(xmin, xmax, dqx), np.arange(ymin, ymax, dqy)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.vstack((xx.flatten(), yy.flatten())).T
    
    # include point in qlist if the point is in the polygon
    inpath = polypath.contains_points(xxyy + 1e-4, radius = 1e-4) # because points on the boundry are not recognized
    return xxyy[inpath, :]

def FindLatticeAcc(q, inFreq = False):
    """Given q, a point in k-space, returns the required frequency setting in Cicero.
    
    (i.e. decompose q in terms of kB12 & kB23, and convert it to corresponding frequency)
    Here q is normalized by K, e.g. FindLatticeAcc(0.8*Mp/K)
    """
    if inFreq: C = 0. # Figure out the correctd parameters for freq_K
    else: C = 1
    e12, e23 = E9c.kB12 / np.linalg.norm(E9c.kB12), E9c.kB23 / np.linalg.norm(E9c.kB23)
    return C * np.linalg.inv(np.transpose([e12, e23])) @ q

# Get rid of this
def InJ(point, J = E9c.Jset1064):
    """Returns 1 if point is in J, 0 if not.
    
    point is a 2-element sequence (usually array); return Boolean as 0 or 1"""
    return int(tuple(point) in J)

def FindH(q, num, Exp_lib, center = (0, 0)):
    """Generates non-interacting Hamiltonian for the specified experiment parameters.
    
    index convention: for m * size + n > 0, m * size + n = index; for <0, size**2 + m * size + n = index
    mm and nn are m' and n' in thesis; q is q_tilde := q/k
    If there is a key in Exp_lib called 'ABoffset' that gives the offset between A and B sites, then that will also
    be added in the Hamiltonian. In our experiment this doesn't exceed 2.3% of V1064, but this can be set to any value.
    Note that ABoffset should already take all the numerical prefactors into consideration. Physically, this means
    ABoffset = 0.011586 * V1064 / 9 * cos(theta), where theta is the angle between magnetic field and z axis.
    """
    def GetPhi(j, l):
        """Handles 1064 superlattice phases that controls relative displacement."""
        if j == 0 and l == 1:
            return -phi12 + phi23
        elif j == 0 and l == -1:
            return phi12 - phi23
        elif j == 1 and l == 0:
            return -phi23
        elif j == -1 and l == 0:
            return phi23
        elif j == 1 and l == 1:
            return phi12
        elif j == -1 and l == -1:
            return -phi12
        else:
            return 0
    
    def get_rel_V(j, l):
        """Returns the relative potential depth for the given j and l (i.e. given beam pair)."""
        # 1064 lattice
        if abs(j) == 1 and abs(l) == 1:
            return B1_rel_E_1064
        elif abs(j) == 1 and abs(l) == 0:
            return B3_rel_E_1064
        elif abs(j) == 0 and abs(l) == 1:
            return B1_rel_E_1064 * B3_rel_E_1064
        # 532 lattice
        elif abs(j) == 2 and abs(l) == 2:
            return B1_rel_E_532
        elif abs(j) == 2 and abs(l) == 0:
            return B3_rel_E_532
        elif abs(j) == 0 and abs(l) == 2:
            return B1_rel_E_532 * B3_rel_E_532
        else:
            return 0
    
    l_unit = Exp_lib["units_dict"]["l_unit"]
    V532, V1064 = Exp_lib['V532'], Exp_lib['V1064']
    B1_rel_E_532, B3_rel_E_532 = np.sqrt(Exp_lib['B1_rel_int_532']), np.sqrt(Exp_lib['B3_rel_int_532'])
    B1_rel_E_1064, B3_rel_E_1064 = np.sqrt(Exp_lib['B1_rel_int_1064']), np.sqrt(Exp_lib['B3_rel_int_1064'])
    phi12, phi23 = Exp_lib['phi12'], Exp_lib['phi23']
    ABoffset1064 = Exp_lib.get('ABoffset1064', False) # ABoffset will be evaluated only if it gives a non-zero value
    size = 2 * num + 1
    H = np.zeros((size**2, size**2), dtype = np.cdouble)
    dg1 = center[0]
    dg2 = center[1]
    
    K = E9c.k_lw * l_unit
    for m in range(-num + dg1, num + dg1 + 1):
        for n in range(-num + dg2, num + dg2 + 1):
            for mm in range(-num + dg1, num + dg1 + 1):
                for nn in range(-num + dg2, num + dg2 + 1):
                    ind0, ind1 = m * size + n, mm * size + nn
                    is_0th_order = int((m == mm) and (n == nn))
                    phi = GetPhi(mm - m, nn - n)
                    rV = get_rel_V(mm - m, nn - n)

                    H[ind0, ind1] += (K**2 / 2. * np.linalg.norm(m * E9c.g1g + n * E9c.g2g + q)**2) * is_0th_order
                    if V1064:
                        # This part were rewritten with a bunch of if statements to include phi's
                        H[ind0, ind1] += rV * (-V1064 * (2/3.) * is_0th_order - (-V1064 / 9.) * np.exp(1j * phi) * InJ((mm - m, nn - n), E9c.Jset1064))
                    if V532:
                        H[ind0, ind1] += rV * (V532 * (2/3.) * is_0th_order - (V532 / 9.) * InJ((mm - m, nn - n), E9c.Jset532))
                    if ABoffset1064:
                        H[ind0, ind1] += rV * (ABoffset1064 * 1j * np.exp(1j * phi) * InJ((mm - m, nn - n), E9c.J1set1064))
                        H[ind0, ind1] += rV * (ABoffset1064 * (-1j) * np.exp(1j * phi) * InJ((mm - m, nn - n), E9c.J2set1064))
    return H

def find_H(q, Exp_lib, Hq_mmat, Hq_nmat, H_lat):
    l_unit = Exp_lib["units_dict"]["l_unit"]
    K = E9c.k_lw * l_unit
    # Tq = K**2 / 2. * np.linalg.norm(np.einsum("i,jk->ijk", E9c.g1g, Hq_mmat)
    #                                 + np.einsum("i,jk->ijk", E9c.g2g, Hq_nmat)
    #                                 + q[:, np.newaxis, np.newaxis], axis = 0)**2
    Tq = K**2 / 2. * np.linalg.norm(np.outer(E9c.g1g, Hq_mmat.diagonal()) + np.outer(E9c.g2g, Hq_nmat.diagonal()) + q[:, np.newaxis], axis = 0)**2
    return np.diag(Tq) + H_lat

def find_H_components(num, Exp_lib, center = (0, 0)):
    """Generates 'components' that make up non-interacting Hamiltonian.
    
    This is based on the observation that the Hamiltonian can be written as a sum of components
    H = T(q) * Hq + H_lat, where
        Hq      is the kinetic energy term, and the only term with a coefficient that depends on q,
        H_lat   is the lattice potential term, and is constant for a given lattice.
    H_lat can be written as a sum of terms
    H_lat = V1064 * (H1064 + ABoffset1064 * H1064AB) + V532 * H532
        Hxxx    is the term from the xxx nm lattice,
        H1064AB is the term that contains the AB offset
    These matrices are generated using for loops and is therefore very costly.
    """
    def GetPhi(j, l):
        """Handles 1064 superlattice phases that controls relative displacement."""
        if j == 0 and l == 1:
            return -phi12 + phi23
        elif j == 0 and l == -1:
            return phi12 - phi23
        elif j == 1 and l == 0:
            return -phi23
        elif j == -1 and l == 0:
            return phi23
        elif j == 1 and l == 1:
            return phi12
        elif j == -1 and l == -1:
            return -phi12
        else:
            return 0
    
    def get_rel_V(j, l):
        """Returns the relative potential depth for the given j and l (i.e. given beam pair)."""
        # 1064 lattice
        if abs(j) == 1 and abs(l) == 1:
            return B1_rel_E_1064
        elif abs(j) == 1 and abs(l) == 0:
            return B3_rel_E_1064
        elif abs(j) == 0 and abs(l) == 1:
            return B1_rel_E_1064 * B3_rel_E_1064
        # 532 lattice
        elif abs(j) == 2 and abs(l) == 2:
            return B1_rel_E_532
        elif abs(j) == 2 and abs(l) == 0:
            return B3_rel_E_532
        elif abs(j) == 0 and abs(l) == 2:
            return B1_rel_E_532 * B3_rel_E_532
        else:
            return 0
    
    V532, V1064 = Exp_lib['V532'], Exp_lib['V1064']
    B1_rel_E_532, B3_rel_E_532 = np.sqrt(Exp_lib['B1_rel_int_532']), np.sqrt(Exp_lib['B3_rel_int_532'])
    B1_rel_E_1064, B3_rel_E_1064 = np.sqrt(Exp_lib['B1_rel_int_1064']), np.sqrt(Exp_lib['B3_rel_int_1064'])
    phi12, phi23 = Exp_lib['phi12'], Exp_lib['phi23']
    ABoffset1064 = Exp_lib.get('ABoffset1064', False) # ABoffset will be evaluated only if it gives a non-zero value
    size = 2 * num + 1
    Hq_mmat = np.zeros((size**2, size**2))
    Hq_nmat = np.zeros((size**2, size**2))
    H_lat   = np.zeros((size**2, size**2), dtype = np.cdouble)
    dg1 = center[0]
    dg2 = center[1]
    
    for m in range(-num + dg1, num + dg1 + 1):
        for n in range(-num + dg2, num + dg2 + 1):
            for mm in range(-num + dg1, num + dg1 + 1):
                for nn in range(-num + dg2, num + dg2 + 1):
                    ind0, ind1 = m * size + n, mm * size + nn
                    is_0th_order = int((m == mm) and (n == nn))
                    phi = GetPhi(mm - m, nn - n)
                    rV = get_rel_V(mm - m, nn - n)

                    Hq_mmat[ind0, ind1] = m * is_0th_order
                    Hq_nmat[ind0, ind1] = n * is_0th_order
                    
                    if V1064:
                        H_lat[ind0, ind1] += rV * (-V1064 * (2/3.) * is_0th_order - (-V1064 / 9.) * np.exp(1j * phi) * InJ((mm - m, nn - n), E9c.Jset1064))
                        if ABoffset1064:
                            H_lat[ind0, ind1] += rV * (ABoffset1064 * 1j * np.exp(1j * phi) * InJ((mm - m, nn - n), E9c.J1set1064))
                            H_lat[ind0, ind1] += rV * (ABoffset1064 * (-1j) * np.exp(1j * phi) * InJ((mm - m, nn - n), E9c.J2set1064))
                    if V532:
                        H_lat[ind0, ind1] += rV * (V532 * (2/3.) * is_0th_order - (V532 / 9.) * InJ((mm - m, nn - n), E9c.Jset532))
    return Hq_mmat, Hq_nmat, H_lat

def FindInteractionIndices(num, center = (0, 0)):
    """Find (m, n) indices involved in nonlinear eigenstate calculation.
    
    These also gives the indices used in rho terms in Bogoliubov calculation.
    """
    index_list = []
    dg1 = center[0]
    dg2 = center[1]
    
    for m in range(-num + dg1, num + dg1 + 1):
        for n in range(-num + dg2, num + dg2 + 1):
            for M in range(-num + dg1, num + dg1 + 1):
                for N in range(-num + dg2, num + dg2 + 1):
                    for mm in range(-num + dg1, num + dg1 + 1):
                        for nn in range(-num + dg2, num + dg2 + 1):
                            # every pair of (m,mm,M) uniquely defines a m_tilde
                            m_t = mm + M - m
                            n_t = nn + N - n
                            if m_t in range(-num + dg1, num + dg1 + 1) and n_t in range(-num + dg2, num + dg2 + 1):
                                # Assume that ignored terms are small; these include c outside of the k space considered
                                index_list.append((m, n, M, N, mm, nn, m_t, n_t))
    return index_list

def FindRhoTildeIndices(num, center = (0, 0)):
    """Find (m, n) indices involved in rho_tilde terms in Bogoliubov calculation."""
    index_list = []
    dg1 = center[0]
    dg2 = center[1]
    
    for m in range(-num + dg1, num + dg1 + 1):
        for n in range(-num + dg2, num + dg2 + 1):
            for M in range(-num + dg1, num + dg1 + 1):
                for N in range(-num + dg2, num + dg2 + 1):
                    for mm in range(-num + dg1, num + dg1 + 1):
                        for nn in range(-num + dg2, num + dg2 + 1):
                            # The only difference is the sign before M & N
                            m_t = mm - M + m # mm - M - m
                            n_t = nn - N + n # nn - N - n
                            if m_t in range(-num + dg1, num + dg1 + 1) and n_t in range(-num + dg2, num + dg2 + 1):
                                # Assume that ignored terms are small; these include c outside of the k space considered
                                index_list.append((m, n, M, N, mm, nn, m_t, n_t))
    return index_list

def AddInteraction(H0, psi, indices, Exp_lib):
    """Add terms arising from interaction to the Hamiltonian for nonlinear eigenstate calculation.
    
    The interaction indices describes how atoms from two initial plane wave components are scattered to two final
    plane wave components. They are the same given some momentum space sizes so is only calculated once (using 
    FindInteractionIndices)."""
    n0 = Exp_lib['n0']
    size = int(np.sqrt(len(H0))) # len(H0) just gives the size of the first dimension
    
    H = copy.deepcopy(H0) # or H0.copy(); also H0.view() creates a shallow copy, and assignment creates a reference
    for (m, n, M, N, mm, nn, m_t, n_t) in indices:
        H[m * size + n, mm * size + nn] += E9c.U_GPE_Rb87 * n0 * psi[m_t * size + n_t] * np.conj(psi[M * size + N])
    
    return H

def FindH_BdG_MFpart(psi, rho_indices, rho_tilde_indices):
    """Find the mean field part of Bogoliubov-de Gennes Hamiltonian.
    
    psi: self-consistent solution of GPE at quasomomentum k
    rho_indices and rho_tilde_indices are given by FindInteractionIndices and FindRhoTildeIndices, respectively.
    The matrix looks like [[h, Delta], [Delta*, h*]]; see my notes
    
    For non-interacting Hamiltonians with size (n, n), this will give a mean field Hamiltonian with size (2n, 2n).
    Note that this doesn't include the non-interacting part of the Hamiltonian in h. Consider a given psi. We want
    to calculate the Bogoliubov spectrum as a function of excitation quasimomentum q about this state, but the only
    q-dependent part of the Hamiltonian is in the non-interacting part of h. The rest is determined solely by psi
    , and also time comsuming to calculate. Therefore, we calculate the part determined by psi, and then add the
    non-interacting part separately with FindH. To complete this Hamiltonian, you need something like
        H_BdG = FindH_BdG_MFpart(refstate, rho_indices, rho_tilde_indices)
        H_BdG[:size**2, :size**2] += FindH(k + q, refstate.num, Exp_lib, refstate.center)
        H_BdG[size**2:, size**2:] += FindH(k + q, refstate.num, Exp_lib, refstate.center)
    Here the quasimomentum used when calculating H0 should be given by (k + q), where k is the quasimomentum of 
    psi (i.e. psi.q), and q is the quasimomentum of the Bogoliubov excitation with respect to k.
    As a reminder, usually we want to calculate the eigensystem of (\tau_x @ H_BdG)."""
    Exp_lib = psi.param
    n0 = Exp_lib['n0']
    size = 2 * psi.num + 1
    
    # h = np.zeros((size**2, size**2), dtype = np.cdouble)
    h = - psi.E * np.identity(size**2, dtype = np.cdouble)
    Delta = np.zeros((size**2, size**2), dtype = np.cdouble)
    for (m, n, M, N, mm, nn, m_t, n_t) in rho_indices:
        h[m * size + n, mm * size + nn] += 2 * E9c.U_GPE_Rb87 * n0 * psi[m_t * size + n_t] * np.conj(psi[M * size + N])
    for (m, n, M, N, mm, nn, m_t, n_t) in rho_tilde_indices:
        Delta[m * size + n, mm * size + nn] += E9c.U_GPE_Rb87 * n0 * psi[m_t * size + n_t] * psi[M * size + N]
    
    return np.block([[h, Delta], [Delta.conj(), h.conj()]])

def GPEFunctional(psi, indices, Exp_lib, q = np.array([0, 0]), center = (0, 0), g = E9c.U_GPE_Rb87):
    """Gross-Pitaevskii energy functional to be minimized."""
    V532, V1064, n0 = Exp_lib['V532'], Exp_lib['V1064'], Exp_lib['n0']
    size = int(np.sqrt(len(psi)))
    num = int((size - 1) / 2)
    dg1 = center[0]
    dg2 = center[1]
    if type(psi) == blochstate:
        q = psi.q
        center = psi.center
    
    Energy = 0
    for m in range(-num + dg1, num + dg1 + 1):
        for n in range(-num + dg2, num + dg2 + 1):
            # Kinetic energy
            Energy += (E9c.k_lw**2 / 2) * psi[m * size + n]**2 * np.linalg.norm(q + m * E9c.G1G + n * E9c.G2G)**2
            
            # Potential energy
            for (j, l) in list(E9c.Jset532):
                if j == 0 and l == 0:
                    Energy += 2/3 * V532 * psi[m * size + n]**2
                else:
                    if m + j in range(-num + dg1, num + dg1 + 1) and n + l in range(-num + dg2, num + dg2 + 1):
                        Energy -= 1/9 * V532 * psi[m * size + n] * psi[(m + j) * size + (n + l)]
            for (j, l) in list(E9c.Jset1064):
                if j == 0 and l == 0:
                    Energy += 2/3 * (-V1064) * psi[m * size + n]**2
                else:
                    if m + j in range(-num + dg1, num + dg1 + 1) and n + l in range(-num + dg2, num + dg2 + 1):
                        Energy -= 1/9 * (-V1064) * psi[m * size + n] * psi[(m + j) * size + (n + l)]
    
    for (m, n, M, N, mm, nn, m_t, n_t) in indices:
        # Interaction energy
        Energy += psi[m * size + n] * psi[M * size + N] * psi[mm * size + nn] * psi[m_t * size + n_t] * (n0 * g / 2)
    return Energy

def GPEResidual(psi_and_mu, H_bare, indices, Exp_lib, g = E9c.U_GPE_Rb87):
    """Calculate the "residual" of GPE, namely, (lhs) - (rhs) of GPE, and return as an array.
    
    Each (but the last) entry of the output array is the residual of GPE (i.e. LHS - RHS), while the last entry
    enforces normalization condition."""
    psi = psi_and_mu[:-1]
    mu = psi_and_mu[-1]
    H = AddInteraction(H_bare, psi, indices, Exp_lib)
    normalization = np.linalg.norm(abs(psi)) - 1
    return np.concatenate((H @ psi - mu * psi, np.array([normalization])))

def GPEResidual2(psi_and_k, mu, indices, Exp_lib, center = (0, 0), g = E9c.U_GPE_Rb87):
    """Calculate the "residual" of GPE, namely, (lhs) - (rhs) of GPE, and return as an array.
    
    Here the last index of psi_and_k is the quasimomentum of the interested state, assumed to be sitting along Kp - Gp.
    This should be changed. This is the correct residual function to use when walking along E instead of k axis."""
    V532, V1064 = Exp_lib['V532'], Exp_lib['V1064']
    psi = psi_and_k[:-1]
    k_in = psi_and_k[-1]
    num = int((int(np.sqrt(len(psi))) - 1) / 2)
    H_bare = FindH(k_in * E9c.kp/E9c.k_sw, num, center, V532, V1064)
    H = AddInteraction(H_bare, psi, indices, Exp_lib)
    normalization = np.linalg.norm(abs(psi)) - 1
    return np.concatenate((H @ psi - mu * psi, np.array([normalization])))

def CheckError(v1, v2, tolerance = 2.5e-4, fail = 1):
    """Check if two unit vectors v1 and v2 are the same within some tolerance."""
    error = 1 - abs(np.vdot(v1,v2)) ** 2
    if error < tolerance:
        return 1, error
    elif error > fail:
        return -1, error # Also a true value, but can be used as a sign of error
    else:
        return 0, error

def FindEigenStuff(H, band, num = 5):
    """Obsolete weaker version of eigh that also checks hermiticity.
    
    band is the range of bands interested in (for example, (0,3) means ground band to 4th band (3rd excited band))
    *** Note that the 4th band is included, so the number of bands returned are (max - min + 1). ***
    when there is only one band of interested, either integer input or (band, band) is accepted"""
    if type(band) == int:
        band = (band, band)
    # Assume Hermitian Hamiltonian
    if not util.IsHermitian(H):
        print("warning: input to eigh is not hermitian. results are wrong")
    return eigh(H, eigvals = band)

def CheckSelfConv(H_bare, state_i, indices, Exp_lib):
    """Check if a state is self-consistent for GPE.
    
    (still using array for state_i in my code, should be cleaned up)
    The function returns (for the most similar e-state) i) energy (E); ii) band number (N); iii) error
    "band number" mentioned above is ranked by eigenvalues of the Hamiltonian generated by state_i, and is not
    necessarily equal to state_i.N. (meaning of N is not super clear in the interacting case as we know it, and I
    usually set N to be the band number of the initial state I used in calculation."""
    # H_bare = FindH(state_i.q, state_i.num, state_i.param, center = state_i.center)
    H_GP = AddInteraction(H_bare, state_i, indices, Exp_lib)
    _conv = np.zeros(10)
    error_temp = np.zeros(10)
    values_temp, states_temp = FindEigenStuff(H_GP, (0, 9))
    for i in range(10):
        _conv[i], error_temp[i] = CheckError(state_i, states_temp[:,i])
    esort = error_temp.argsort()
    bandnum = esort[0]#int(error_temp.argmin())
    #print("(second smallest error: N = {}, E = {}, error = {})".format(esort[1], values_temp[esort[1]], error_temp[esort[1]]))
    return values_temp[bandnum], bandnum, error_temp.min()

def FindGroundState(H_bare, state_i, indices):
    # use imaginary time method to find the g.s. of a given q
    # here "error" is defined as before; this is different from what people usually do, i.e. monitor chemical potential
    tolerance = 1e-12
    error = 1
    dt = 0.01
    steps = 1
    state_previous = state_i
    while error > tolerance and steps < 1000:
        H_GP = AddInteraction(H_bare, state_i, indices)
        state_new = expm(-H_GP * dt) @ state_previous
        state_new = util.Normalize(state_new)
        _, error = CheckError(state_new, state_previous)
        state_previous = state_new
        if steps % 50 == 0: print("loop #{}: error = {}".format(steps, error))
        steps += 1
    
    H_GP = AddInteraction(H_bare, state_i, indices)
    tempE, tempN, temperror = CheckSelfConv(H_GP, state_new, indices)
    tempbloch = blochstate(state_new, E = tempE, q = state_i.q, N = tempN, error = temperror, info = "From imaginary time propagation")
    return tempbloch

def Predictor(pstate, ppstate, step, model = 'taylor1'):
    # determines the next initial guess
    # pstate & ppstate are previous state and the state previous to that, respectively
    if model == 'taylor1':
        # predict by linear extrapolation; only take one continuation parameter
        # Note that the bloch state parameters are inherited from pstate
        if np.allclose(pstate.q, ppstate.q):
            r = 0
        else:
            r = step / (pstate.q[1] - ppstate.q[1])
        return util.Normalize((1 + r) * pstate - r * ppstate), (1 + r) * pstate.E - r * ppstate.E
    elif model == 'taylor1E':
        # predict by linear extrapolation, but using E as variable
        if np.allclose(pstate.E, ppstate.E):
            r = 0
        else:
            r = step / (pstate.E - ppstate.E)
        return util.Normalize((1 + r) * pstate - r * ppstate), (1 + r) * pstate.q[1] - r * ppstate.q[1]
    else:
        print("undefined model type")
        raise

#%% Probably not used anywhere, but included for historical reason
class LinSupState:
    """Linear superposition of different blochstates, possibly at different q or in different bands.
    
    state_list: a list of blochstates not necessarily linearly independent of one another
    weight_list: a list of corresponding weights, complex in general
    This class is not used very often, so I didn't bother to refine the code. For example, FindDensity is better
    defined as a function that calls object.find_density for any object."""
    def __init__(self, state_list, weight_list):
        if len(state_list) != len(weight_list): raise BaseException("Lengths of state and weight lists don't match")
        for s in state_list:
            if type(s) != blochstate: raise BaseException("LinSupState only deal with blochstates")
        self.states = state_list
        self.weights = weight_list
        self.qs = [s.q for s in state_list]
        # Figure out normalization: if there are states with the same q the plane waves should be added together
        # I am being lazy and decided to assume that all states have the same center and different q
        norm_factor = 0        
        for s, w in zip(state_list, weight_list):
            norm_factor += np.linalg.norm(s)**2 * abs(w)**2
        self.norm_factor = np.sqrt(norm_factor)
        
    def totext(self):
        '''Returns a human readable string that focuses on blochstate labels, i.e. |q, n>.'''
        substr = []
        for s, w in zip(self.states, self.weights):
            warg, wph = float(abs(w) / self.norm_factor), np.angle(w)
            substr.append("{:.4} exp({:.2} *2pi){}".format(warg, wph / 2 / np.pi, s.totext()))
        return " + ".join(substr)
            
    def findj(self, x, y):
        '''Finds the current density.'''
        raise BaseException("don't know how to implement yet") # probably not hard, just annoying...