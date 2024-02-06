import numpy as np
import matplotlib.pyplot as plt

from E9_fn.E9_constants import *

def PPart(metal, T):
    """[Pa] Returns the partial pressure of some metal at temperature T.
    
    The formula is given by
                PPart [mbar] = (T <= Tm, solid) 10**(cs - ds / T)
                               (T > Tm, liquid) 10**(cl - dl / T)
    Note that the values from various references might need to be adjusted to account for unit conversion.
    torr -> mbar: -0.125; atm -> mbar: +3.006; mbar -> Pa: """
    # pairs of "metal": (cs, ds, cl, dl, Tm)
    cdLib = {"K": (7.9667, 4646, 7.4077, 4453, 336.8), "Rb": (7.863, 4215, 7.318, 4040, 312.45), \
             "Na": (8.304, 5603, 7.71, 5377, 370.95)}
    cs, ds, cl, dl, Tm = cdLib[metal]
    if T <= Tm:
        return 10**(cs - ds / T) * 100
    else:
        return 10**(cl - dl / T) * 100

def PlotPpartInAlloy(alloy, Ts = [300, 400], xA = (0, 1), ratio_in = 'liquid'):
    """Plot the partial pressure of species in an alloy.
    
    See [Ishikawa17]. Note that this function assumes that the system is in the liquid - gas phase.
        alloy: string of the alloy of interest. e.g. "KRb" , "CsNa" ... (in alphabetical order; must be in RTlLib)
        T: [K] temperature. Either a value or a list.
        ratio_in: either "liquid" or "gas" . For "liquid" , xA = xA in the paper; for "gas" , xA = yA."""
    if ratio_in != "liquid":
        print("not implemented yet")
        return
    
    def LinearTFn(a, b):
        def fn(T):
            return LinearFn(a, b, T)
        return fn
    def GetdeltaA(l0, l1, l2, xA, xB, xAB):
        return xB * (l0 + l1 * (2 * xAB + 1) + l2 * (3 * xAB + 2) * xAB)
    def GetaA(l0, l1, l2, xA, xB, xAB):
        """Activity"""
        return np.exp((1 - xA) * GetdeltaA(l0, l1, l2, xA, xB, xAB)) * xA
    def GetdeltaB(l0, l1, l2, xA, xB, xAB):
        return xA * (l0 + l1 * (2 * xAB - 1) + l2 * (3 * xAB - 2) * xAB)
    def GetaB(l0, l1, l2, xA, xB, xAB):
        """Activity"""
        return np.exp((1 - xB) * GetdeltaB(l0, l1, l2, xA, xB, xAB)) * xB
    # pairs of "alloy": (RTl0(T), RTl1(T), RTl2(T), pA*, pB*, Astr, Bstr)
    RTlLib = {"KRb": (LinearTFn(501.2, -0.831), LinearTFn(9.789, -0.728), LinearTFn(56.082, -0.16), "K", "Rb"), \
              "NaRb": (LinearTFn(4948.676, 0.341), LinearTFn(1743.399, -1.417), LinearTFn(251.419, 1.887), "Na", "Rb")}
    xA = np.linspace(xA[0], xA[1])
    xB = 1 - xA
    xAB = xA - xB
    
    RTls = RTlLib[alloy]
    Astr, Bstr = RTls[3], RTls[4]
    Tnum = len(Ts)
    f_P = plt.figure(100, figsize = (12, 8))
    f_P.clf()
    f_P.suptitle(alloy + ' alloy')
    for i, T in enumerate(Ts):
        RTl0, RTl1, RTl2 = RTls[0](T), RTls[1](T), RTls[2](T)
        l0, l1, l2 = RTl0 / R_gas / T, RTl1 / R_gas / T, RTl2 / R_gas / T
        aA, aB = GetaA(l0, l1, l2, xA, xB, xAB), GetaB(l0, l1, l2, xA, xB, xAB)
        pAstar, pBstar = PPart(Astr, T), PPart(Bstr, T)
        pA, pB = aA * pAstar, aB * pBstar
        ax_a = f_P.add_subplot(2, Tnum, i + 1)
        ax_a.plot(xA, aA, '-r', label = Astr)
        ax_a.plot(xA, xA, '--r', label = Astr + ' (Ideal)')
        ax_a.plot(xA, aB, '-b', label = Bstr)
        ax_a.plot(xA, 1 - xA, '--b', label = Bstr + ' (Ideal)')
        ax_a.legend()
        ax_a.set_xlabel('xA ({})'.format(Astr))
        ax_a.set_ylabel('Activity')
        ax_a.set_title('T = {} K'.format(T))
        
        ax_P = f_P.add_subplot(2, Tnum, i + 1 + Tnum)
        ax_P.plot(xA, pA, '-r', label = Astr)
        # ax_P.plot(xA, xA, '--r', label = Astr + ' (Ideal)')
        ax_P.plot(xA, pB, '-b', label = Bstr)
        # ax_P.plot(xA, 1 - xA, '--b', label = Bstr + ' (Ideal)')
        ax_P.plot(xA, pA + pB, '-', c = '#f68a1e', label = '$P_{tot}$')
        ax_P.plot(xA, pBstar + xA * (pAstar - pBstar), '--', c = '#f68a1e', label = '$P_{tot}$ (ideal)')
        ax_P.legend()
        ax_P.set_xlabel('xA ({})'.format(Astr))
        ax_P.set_ylabel('P (Pa)')
    f_P.tight_layout()
    return f_P