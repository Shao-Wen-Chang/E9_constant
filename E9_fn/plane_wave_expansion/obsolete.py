# TODO: I want to replace blochstate with a class that describes the experiment instead + a new state class
# Simplifications I want:
#   don't keep k_center anywhere, I never use it
#   stick to SI units?
#   don't keep the bandstart/end thing
class e9_lattice_exp():
    """Optical lattice related parameters and methods."""
    def __init__(self, V1064Er, V532Er
                 , phi12 = 0., phi23 = 0.
                 , B1_rel_int_1064 = 1, B1_rel_int_532 = 1, B3_rel_int_1064 = 1, B3_rel_int_532 = 1
                 , q_num: int = 6):
        self.V1064Er = V1064Er  # lattice depths in units of Er
        self.V532Er = V532Er
        self.phi12 = phi12      # relative phases between 1064 beams (532 phases are set to 0)
        self.phi23 = phi23
        self.B1_rel_int_1064 = B1_rel_int_1064
        self.B1_rel_int_532 = B1_rel_int_532
        self.B3_rel_int_1064 = B3_rel_int_1064
        self.B3_rel_int_532 = B3_rel_int_532

        # non-physical parameters
        self.q_num = q_num      # number of plane waves used in plane wave expansion
    
    def create_new_exp(self, **kwargs):
        """Return a new experiment with parameters in kwargs, but otherwise the same as self."""
        exp_params = self.__dict__
        exp_params.update(kwargs)
        return e9_lattice_exp(**exp_params)
    
    def get_H(self, q: np.ndarray, **kwargs):
        """Get the Hamiltonian of the experiment, possibly with some changes in parameters.
        
        To change parameters, use e.g.
        my_e9_exp.get_H(q_now, phi12 = np.pi * (2/3), phi23 = np.pi * (-2/3))  # decorated triangular lattice
        WARNING: this doesn't change the value of attributes of the original e9_lattice_exp
        object. Use create_new_exp() if a new object with the new parameters should be created.
        
        I didn't merge create_new_exp() into get_H() to avoid generating the helper matrices
        repeatedly, which is expensive.

        TODO: this won't work for relative intensities yet, need to update find_H_components

        Args:
            q:      quasimomentum at which H is evaulated.
        """
        exp_params = self.__dict__
        exp_params.update(kwargs)

        if not hasattr("_Hq_mmat"):
            if np.isclose(self.V1064Er, 0):
                H_1064 = None # maybe have find_H_components return None
            else:
                Hq_mmat, Hq_nmat, H_532, H_1064 = find_H_components(self.q_num, exp_params)
            setattr(self, "_Hq_mmat", Hq_mmat)
            setattr(self, "_Hq_nmat", Hq_nmat)
            setattr(self, "_H_532", H_532)
            setattr(self, "_H_1064", H_1064)
        return find_H(q, exp_params, self._Hq_mmat, self._Hq_nmat, self._H_532, self._H_1064)

# a new class blochstate(): here


def PlotBZ(BZcolor = E9c.BZcolor_PRL, fignum = 100):
    """Convienent function for plotting BZ."""
    fig = plt.figure(fignum)
    fig.clf()
    ax_BZ = fig.add_subplot(111)
    PlotBZSubplot(ax_BZ, BZcolor = BZcolor)
    if type(qset) == str: # see PlotBZSubplot
        qstr = qset
    else:
        qstr = qset[0]
    fig.suptitle('qset: ' + qstr)
    return ax_BZ

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