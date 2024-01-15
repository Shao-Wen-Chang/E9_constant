import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import table as plot_table

from E9_fn import util
import E9_fn.thermodynamics as thmdy
# This file is meant to be a collection of different ways that I might model our experiment:
# e.g. Fermi-Hubbard model with parameters like t, U, V(x), and something that specifies
# connectivity; some DoS with some interaction parameter; Heisenberg model with localized
# spins, etc. Each of them should be specified by a different class.

# Since the definitions depend on the specific use case when these classes are defined and
# I will likely forget about what they're meant for, please add the use case in the comment

#%% DoS model - used in thermodynamical calculations
class DoS_exp():
    '''Models that focuses on the density of states (DoS) of possibly several species.
    
    Each "experiment" or "system" contains one or more atomic species in a finite size
    system. They can be a single isotope in different spin states, several atomic species
    , etc. Each of the species has their own set of properties, such as the size of the
    system, DoS, # of particles, or interaction strength.

    A "reservoir" is a special kind of species that is actually the same species as some
    other species ("system"), but with some different properties, such as a different
    DoS. This is useful when e.g. one consider spinless fermions in a step potential trap.
    The reservoir species will share the chemical potential with the system. They must
    be listed at the end of the tuple.
    
    Currently, the properties shared by all system are:
        1. Temperature (thermodynamical equilibrium);
        2. -'''
    def __init__(self, T: float, species_list: list[dict]):
        '''Each of the elements in species_list must be defined as a dictionary, with
        at least the following key-value pairs:
            "name": str                  # name of the species
            "V": int                     # number of orbitals / size of the system
            "Np": int                    # number of particles (will be overwritten if
                                           "reservoir" is not "", see below)
            "stat": "bose" or "fermi"    # statistics of the particle considered.
            "DoS": callable like f(E)    # density of states
            "E_range": (float, float)    # energies considered in calculation
            "reservoir": str             # "" if not a reservoir; "name" if it acts as the
                                           reservoir of some other species
            "comment": {}                # comments from various functions
        Additional key-value pairs are assigned each time a value is calculated.
        
        Other inputs:
            T: temperature of the system.'''
        self.T = T

        self.species_list = species_list
        self.names = util.all_values_from_key(species_list, "name")
        self.Vs = util.all_values_from_key(species_list, "V")
        self.Nps = util.all_values_from_key(species_list, "Np")
        self.stats = util.all_values_from_key(species_list, "stat")
        self.DoSs = util.all_values_from_key(species_list, "DoS")
        self.E_ranges = np.array(util.all_values_from_key(species_list, "E_range"))
        self.reservoirs = util.all_values_from_key(species_list, "reservoir")

        # the index of system referenced by reservoir, if any
        self._refsys = [None for _ in self.species_list]
        for i, r in enumerate(self.reservoirs):
            if r:
                try:
                    self._refsys[i] = self.names.index(r)
                except ValueError:
                    print(r + " not found in the list of names")

        # self.comments = util.all_values_from_key(species_list, "comment") # not useful if not updated
    
    def check_consistency(self, update: bool = True):
        # '''Check that all the attributes of the object matches those in self.species_list.
        
        # Currently this method assumes that the species in species_list all have the same
        # set of keys. This is desired anyway.
        #     update: if True, then the attribute values are updated if there are any
        #             inconsistencies. If False, only prints a warning.'''
        # all_keys = self.species_list[0].keys()
        # for key in all_keys:
        #     attr_name = key + 's' # This is bad coding practice
        #     for i, sp in enumerate(self.species_list):
        #         value_in_attr = eval("self.{}".format(attr_name))[i]
        #         if value_in_attr != sp[key]:
        #             print("Inconsistency found in species #{}:".format(i))
        #             if update: 
        #                 print("Replaced self.{}[{}] = {} with value {}".format(
        #                     attr_name, i, value_in_attr, sp[key]))
        #                 eval("self.{}".format(attr_name))[i] = sp[key]
        pass

    ### Thermodynamical calculations
    def find_outputs(self) -> None:
        '''Helper function that calculates everything I can calculate.'''
        self.find_E_orbs()
        self.find_mus()
        self.find_Es()
        self.find_Ss()
    
    def find_E_orbs(self) -> None:
        '''Calculate the list of energies sampled from each DoS for each species.
        
        This method both modifies the entries in species_list, and add the parameter E_orbss to
        itself. The behavior of other find_xxx functions are similar.'''
        for sp in self.species_list:
            sp["E_orbs"] = thmdy.E_orbs_from_DoS(sp["DoS"], sp["E_range"], sp["V"])
        self.E_orbss = util.all_values_from_key(self.species_list, "E_orbs")
    
    def find_mus(self) -> None:
        '''Calculate the chemical potential for each species.
        
        For reservoirs, mu is determined by the referenced system, and this function finds
        Np instead.
        N_BEC = 0 always for fermionic species.'''
        for i, sp in enumerate(self.species_list):
            if not sp["reservoir"]:
                # Not a reservoir - find chemical potential by matching particle number
                sp["mu"], sp["N_BEC"] = thmdy.find_mu(sp["E_orbs"], self.T, sp["Np"], sp["stat"])
            else:
                # Is reservoir - find the chemical potential to match to, then update
                # the particle number in reservoir
                sp["mu"] = self.species_list[self._refsys[i]]["mu"]
                sp["Np"] = thmdy.find_Np(sp["E_orbs"], self.T, sp["mu"], sp["stat"])
                sp["N_BEC"] = 0 # Not implemented yet
                sp["comment"]["find_mus"] = \
                "Finding N_BEC for bosonic systems with reservoirs are not implemented yet"
        
        self.mus = util.all_values_from_key(self.species_list, "mu")
        self.Nps = util.all_values_from_key(self.species_list, "Np")
        self.N_BECs = util.all_values_from_key(self.species_list, "N_BEC")
        self.comments = util.all_values_from_key(self.species_list, "comment")


    def find_Es(self) -> None:
        '''Calculate the energy for each species.'''
        for sp in self.species_list:
            sp["E"] = thmdy.find_E(sp["E_orbs"], self.T, sp["mu"], sp["stat"], sp["N_BEC"])
        self.Es = util.all_values_from_key(self.species_list, "E")

    def find_Ss(self) -> None:
        '''Calculate the entropy for each species.'''
        for sp in self.species_list:
            sp["S"] = thmdy.find_S(sp["E_orbs"], self.T, sp["Np"], sp["mu"], sp["E"], sp["stat"], sp["N_BEC"])
        self.Ss = util.all_values_from_key(self.species_list, "S")
    
    ### plot related methods
    def plot_DoSs(self, ax = None):
        '''Visulaization of DoS + filling.'''
        if ax is None:
            fig_DoS = plt.figure(1)
            fig_DoS.clf()
            ax_DoS = fig_DoS.add_subplot(111)
        else:
            ax_DoS = ax

        # Plot DoS and filling
        max_DoS = np.zeros(len(self.species_list))
        for i, sp in enumerate(self.species_list):
            E_orbs_plot = np.linspace(sp["E_range"][0], sp["E_range"][1], 1000)
            if sp["N_BEC"] != 0: E_orbs_plot = E_orbs_plot[1:]
            DoS_values = sp["DoS"](E_orbs_plot)
            max_DoS[i] = max(DoS_values)
            p = ax_DoS.plot(DoS_values, E_orbs_plot, '-', label = sp["name"])
            
            if sp["stat"] == "fermi":
                ax_DoS.fill_betweenx(E_orbs_plot, DoS_values * util.fermi_stat(E_orbs_plot, self.T, sp["mu"])
                                    , '--', alpha = 0.3)
            elif sp["stat"] == "bose":
                ax_DoS.fill_betweenx(E_orbs_plot, DoS_values * util.bose_stat(E_orbs_plot, self.T, sp["mu"])
                                    , '--', alpha = 0.3)
                
            ax_DoS.axhline(sp["mu"], color = p[0].get_color(), ls = '--'
                        , label = r'$\mu = ${:.3f}, $s = ${:.4f}'.format(sp["mu"], sp["S"] / sp["Np"]))
            if sp["N_BEC"] != 0:
                util.plot_delta_fn(ax, x0 = sp["E_orbs"][0], a0 = sp["N_BEC"], a_plt = max_DoS[i] * 0.8,
                                   axis = 'y', color = p[0].get_color(), head_width = 0.5, head_length = 0.3)

        # ax_DoS.set_ylim(min(min(self.mus), E_orbs_plot[0]) - 0.5, max(max(self.mus), E_orbs_plot[-1]) + 0.5)
        ax_DoS.set_xlabel("DoS [arb.]")
        ax_DoS.set_ylabel("E/t")
        ax_DoS.set_title(r'DoS ($T = ${:.2f})'.format(self.T))
        ax_DoS.set_xlim(0, max(max_DoS) * 1.5)
        ax_DoS.legend()
        return ax_DoS
    
    def tabulate_params(self,
                        ax = None,
                        hidden: list[str] = ["DoS", "E_orbs"],
                        str_len: int = 25):
        '''Tabulate all the currently available parameters.
        
        "comment" is a dictionary of str. It is currently handled in an ugly way.
            hidden: a list of keys to ignore.
            str_len: length of the string displayed. Long strings are truncated, and short
                     strings are padded with spaces to the left.'''
        if ax is None:
            fig_table = plt.figure(1)
            fig_table.clf()
            ax_table = fig_table.add_axes(111)
        else:
            ax_table = ax
        
        displayed_keys = list(self.species_list[0].keys())
        for x in hidden: displayed_keys.remove(x)
        all_values_str = [[str(sp[key]).rjust(str_len)[:str_len] for key in displayed_keys]
                          for sp in self.species_list]
        plot_table(ax_table, cellText = list(zip(*all_values_str)),
                   rowLabels = displayed_keys, loc = 'center')
        return ax_table

#%% simulation
if __name__ == "__main__":
    # Moved to projects on 20240114
    pass