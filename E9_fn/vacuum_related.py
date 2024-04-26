import numpy as np
from collections import namedtuple

#%% Constants
T_room          = 273.15 + 22
M_gas           = 28 # Use 2 for H2 (i.e. UHV), 28 for N2
C_func_str      = "C_circular"
S_ion_pump      = {2: 60,
                   28: 25}
S_NEG           = {2: 500,
                   28: 125}
S_gauge         = {2: 0.5,
                   28: 0.5}
S_null          = {2: 0,
                   28: 0}

#%% Basic formulae
def C_add(C_list):
    if np.any(np.array(C_list) == 0):
        return 0
    C_inv = 0
    for C in C_list:
        C_inv += 1 / C
    return 1 / C_inv

def C_tube(D, L, M = M_gas, T = T_room):
    """[Torr-L/s] Conductance of a long tube in the molecular flow regime.
    
    This formula works for long tubes (~20% error at L/D = 10, better if larger)
    D: [cm] inner diameter
    L: [cm] length
    M: [amu] mass of the molecule of concern
    T: [K] temperature"""
    return 3.81 * np.sqrt(T / M) * (D**3 / L)

def C_orfice(A, M = M_gas, T = T_room):
    return 3.624 * np.sqrt(T / M) * A

def C_circular(D, L, M = M_gas, T = T_room):
    """More sophisticated formula."""
    alpha_T = 1.0667 / (1.0667 + (L / D)**0.94629) # Transimission probability
    return alpha_T * C_orfice((D/2)**2 * np.pi, M, T)

def MFP():
    """Mean free path"""
    pass

def C_fitting_sec(fs, C_func = C_tube):
    """[Torr-L/s] Conductance of a fitting_sec.
    
    Ideally this should automatically use the best formula for the given
    dimensions, as I add them here."""
    D, L = fs.ID, fs.length
    return C_func(D, L)
#%% Database
# DNxx: this is the tube ID, more useful for conductance calculations
# "2.75": this is the flange OD, useful for mechanical designs
fitting_sec = namedtuple("fitting_sec", ["name", "ID", "length"])

MOT_nipple_and_reducer  = fitting_sec("MOT_nipple_and_reducer", 3.5, 5.8)
cross_east              = fitting_sec("cross_east", 6.3, 8.62)
cross_west              = fitting_sec("cross_west", 6.3, 8.62)
cross_top               = fitting_sec("cross_top", 3.5, 4.37)
cross_bottom            = fitting_sec("cross_bottom", 3.5, 4.37)
pump_elbow              = fitting_sec("pump_elbow", 6.3, 10.465 * np.pi / 4)
gauge_tee               = fitting_sec("gauge_tee", 3.5, 12.5)
viton_valve             = fitting_sec("viton_valve", 3.5, 12.5) # temporary
NEG_tee                 = fitting_sec("NEG_tee", 3.5, 4.37)

C_all_fittings = {MOT_nipple_and_reducer: 0,
                  cross_east: 0,
                  cross_west: 0,
                  cross_top: 0,
                  cross_bottom: 0,
                  pump_elbow: 0,
                  gauge_tee: 0,
                  viton_valve: 0,
                  NEG_tee: 0,}

for f in C_all_fittings.keys():
    C_all_fittings[f] = C_fitting_sec(f, eval(C_func_str))

#%% calculate conductance
class much_tubes():
    def __init__(self, fitting_list, name: str, S_pump = 0) -> None:
        self.fitting_list = fitting_list
        self.name = name
        self.S_pump = S_pump

        self.C_tot = C_add([C_all_fittings[f] for f in self.fitting_list])
        self.S_reduced = C_add([self.C_tot, self.S_pump[M_gas]])

to_cross_center = much_tubes([MOT_nipple_and_reducer,
                              cross_east], "cross center", S_null)

to_ion_pump = much_tubes([cross_west,
                          pump_elbow], "ion pump", S_ion_pump)

to_NEG = much_tubes([cross_bottom,
                     viton_valve,
                     NEG_tee], "NEG (temp)", S_NEG)

to_gauge = much_tubes([cross_top,
                       gauge_tee], "gauge", S_gauge)

#%% main
def main():
    print("Using T = {}, M = {}; conductance formula = {}".format(T_room, M_gas, C_func_str))
    print("")
    for f in C_all_fittings.keys():
        print("C_{} = {:.2f}".format(f.name, C_all_fittings[f]))
    for t in [to_ion_pump, to_NEG, to_gauge]:
        print("")
        print("Conductance from the center of the cross to {} = {}".format(t.name, t.C_tot))
        print("pump speed of {} at the reducer:".format(t.name))
        print("    bare pump speed = {}".format(t.S_pump[M_gas]))
        print("    reduced speed = {:.2f}".format(t.S_reduced))
    print("")
    print("Conductance from the center of the cross to the main chamber = {:.2f}".format(
        to_cross_center.C_tot))
    print("Total pump speed with ion pump only = {:.2f}".format(
        C_add([to_ion_pump.S_reduced, to_cross_center.C_tot])))
    print("Total pump speed with ion pump + NEG = {:.2f}".format(
        C_add([to_ion_pump.S_reduced + to_NEG.S_reduced, to_cross_center.C_tot])))
    
if __name__ == "__main__":
    main()