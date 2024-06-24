import numpy as np
import copy
import pickle
import sys
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
        https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array"""
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
        (should be convenient for e.g. getting q = -1 from q = 1)"""
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
        
        Should combine with findvy when I care enough."""
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
        return self.q[0] + (float((np.roll(p, 2 * self.num * (self.num - dg1 + 1) - (dg1 + dg2)).reshape(self.ksize, self.ksize) * vxmask).sum()) / k)
        
    def findvy(self):
        """finds the group velocity in y direction (normalized s.t. kp = 1)"""
        p = self.population()
        dg1 = p.center[0]
        dg2 = p.center[1]
        g1y = E9c.g1[1]
        g2y = E9c.g2[1]
        mg = np.mgrid[(-p.num + dg1):(p.num + dg1 + 1), (-p.num + dg2):(p.num + dg2 + 1)]
        vymask = mg[0] * g1y + mg[1] * g2y
        return self.q[1] + (float((np.roll(p, 2 * self.num * (self.num - dg1 + 1) - (dg1 + dg2)).reshape(self.ksize, self.ksize) * vymask).sum()) / k)
    
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
        
        I decided to not modify __str__ because I sometimes still print the whole state out to check its elements."""
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
            substr.append("{:.4} exp({:.2} *2pi){}".format(warg, wph / 2 / pi, s.totext()))
        return " + ".join(substr)
            
    def findj(self, x, y):
        '''Finds the current density.'''
        raise BaseException("don't know how to implement yet") # probably not hard, just annoying...