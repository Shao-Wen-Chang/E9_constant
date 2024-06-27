# Recommended import call: import E9_fn.E9_models as E9M
import logging
from collections import defaultdict, namedtuple
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

#%% DoS models - used in thermodynamical calculations
# These are models that focuses on the density of states (DoS) of possibly several species.
# My terminologies:
# Each "experiment" contains one or more "subregion" in a finite size system.
# Each "subregion" can be a single isotope in different spin states,
# several atomic species, or even the same spin state but with different DoS.
# They are defined by their own set of properties, such as the size of the
# system, DoS, # of particles, or interaction strength.
# Two subregion are of the same "species" if they are indistinguishable particles..

# A "reservoir" is a special kind of subregion that is actually the same species as some
# other species ("system"), but with some different properties, usually a different
# DoS. This is useful when e.g. one consider spinless fermions in a step potential trap.
# The reservoir subregion will share the chemical potential with the system.

class NVT_exp():
    """Canonical ensemble (fix (N, V, T)).
    
    This ensemble has proven to be pretty inconvenient when one needs a fixed total particle number
    and entropy."""
    def __init__(self, T: float, subregion_list: list[dict]):
        """Each of the elements in subregion_list must be defined as a dictionary, with
        at least the following key-value pairs:
            "name": str                  # name of the subregion
            "V": int                     # number of orbitals / size of the system
            "Np": int                    # number of particles (will be overwritten if
                                           "reservoir" is not "", see below)
            "stat": 1 or -1              # 1 for fermions, -1 for bosons
            "DoS": callable like f(E)    # density of states
            "E_range": (float, float)    # energies considered in calculation
            "reservoir": str             # "" if not a reservoir; "name" if it acts as the
                                           reservoir of some other subregion
            "comment": {}                # comments from various functions
        Additional key-value pairs are assigned each time a value is calculated.
        
        Other Args:
            T: temperature of the system."""
        self.T = T

        self.subregion_list = subregion_list
        self.names = util.all_values_from_key(subregion_list, "name")
        self.Vs = util.all_values_from_key(subregion_list, "V")
        self.Nps = util.all_values_from_key(subregion_list, "Np")
        self.stats = util.all_values_from_key(subregion_list, "stat")
        self.DoSs = util.all_values_from_key(subregion_list, "DoS")
        self.E_ranges = np.array(util.all_values_from_key(subregion_list, "E_range"))
        self.reservoirs = util.all_values_from_key(subregion_list, "reservoir")

        # the index of system referenced by reservoir, if any
        self._refsys = [None for _ in self.subregion_list]
        for i, r in enumerate(self.reservoirs):
            if r:
                try:
                    self._refsys[i] = self.names.index(r)
                except ValueError:
                    logging.error(r + " not found in the list of names")

        # self.comments = util.all_values_from_key(subregion_list, "comment") # not useful if not updated

    ### Thermodynamical calculations
    def find_outputs(self) -> None:
        """Helper function that calculates everything I can calculate."""
        self.find_E_orbs()
        self.find_mus()
        self.find_Es()
        self.find_Ss()
    
    def find_E_orbs(self) -> None:
        """Calculate the list of energies sampled from each DoS for each subregion.
        
        This method both modifies the entries in subregion_list, and add the parameter E_orbss to
        itself. The behavior of other find_xxx functions are similar."""
        for sp in self.subregion_list:
            sp["E_orbs"] = thmdy.E_orbs_from_DoS(sp["DoS"], sp["E_range"], sp["V"])
        self.E_orbss = util.all_values_from_key(self.subregion_list, "E_orbs")
    
    def find_mus(self) -> None:
        """Calculate the chemical potential for each subregion.
        
        For reservoirs, mu is determined by the referenced system, and this function finds
        Np instead.
        N_BEC = 0 always for fermionic subregion."""
        for i, sp in enumerate(self.subregion_list):
            if not sp["reservoir"]:
                # Not a reservoir - find chemical potential by matching particle number
                sp["mu"], sp["N_BEC"] = thmdy.find_mu(sp["E_orbs"], self.T, sp["Np"], sp["stat"])
            else:
                # Is reservoir - find the chemical potential to match to, then update
                # the particle number in reservoir
                sp["mu"] = self.subregion_list[self._refsys[i]]["mu"]
                sp["Np"] = thmdy.find_Np(sp["E_orbs"], self.T, sp["mu"], sp["stat"])
                sp["N_BEC"] = 0 # Not implemented yet
                if sp["stat"] == -1: sp["comment"]["find_mus"] = \
                "Finding N_BEC for bosonic systems with reservoirs are not implemented yet"
        
        self.mus = util.all_values_from_key(self.subregion_list, "mu")
        self.Nps = util.all_values_from_key(self.subregion_list, "Np")
        self.N_BECs = util.all_values_from_key(self.subregion_list, "N_BEC")
        self.comments = util.all_values_from_key(self.subregion_list, "comment")


    def find_Es(self) -> None:
        """Calculate the energy for each subregion."""
        for sp in self.subregion_list:
            sp["E"] = thmdy.find_E(sp["E_orbs"], self.T, sp["mu"], sp["stat"], sp["N_BEC"])
        self.Es = util.all_values_from_key(self.subregion_list, "E")

    def find_Ss(self) -> None:
        """Calculate the entropy for each subregion."""
        for sp in self.subregion_list:
            sp["S"] = thmdy.find_S(sp["E_orbs"], self.T, sp["Np"], sp["stat"], sp["mu"], sp["E"], sp["N_BEC"])
        self.Ss = util.all_values_from_key(self.subregion_list, "S")
    
    def find_N_tot(self) -> dict:
        """Find the total number of particles for each subregion by summing over reservoirs.
        
        Returns:
            N_tot: looks like e.g. {"fermi1": 1300, "fermi2": 400, ...}"""
        N_tot = defaultdict(int)
        for sp in self.subregion_list:
            if sp["reservoir"] == "":
                N_tot[sp["name"]] += sp["Np"]
            else:
                N_tot[sp["reservoir"]] += sp["Np"]
        return N_tot

    ### plot related methods
    def plot_DoSs(self, ax = None, offset_traces: bool = False):
        """Visulaization of DoS + filling.
        
        Args:
            offset_traces: if True, offset each DoS trace by a bit horizontally."""
        _, ax_DoS = util.make_simple_axes(ax, fignum = 1)

        # Plot DoS and filling
        max_DoS = np.zeros(len(self.subregion_list))
        for i, sp in enumerate(self.subregion_list):
            # Energies used for plotting is generated separately (less points than
            # E_orbs); ignore the ground state if there's a BEC
            E_orbs_plot = np.linspace(sp["E_range"][0], sp["E_range"][1], 10000)
            if sp["N_BEC"] != 0: E_orbs_plot = E_orbs_plot[1:]
            DoS_values = sp["DoS"](E_orbs_plot)
            max_DoS[i] = max(DoS_values)
            filling = DoS_values * util.part_stat(E_orbs_plot, self.T, sp["mu"], sp["stat"])
            
            # plotting
            off_h = 0
            if offset_traces:
                off_h = 0.1 * i * max_DoS[i]
                ax_DoS.axvline(off_h, color = 'k', ls = '-', lw = 0.5)
            pDoS = ax_DoS.plot(DoS_values + off_h, E_orbs_plot, '-', label = sp["name"])
            ax_DoS.fill_betweenx(E_orbs_plot, x1 = filling + off_h, x2 = off_h
                                 , ls = '--', alpha = 0.3)
            ax_DoS.axhline(sp["mu"], color = pDoS[0].get_color(), ls = '--'
                        , label = r'$\mu = ${:.3f}, $s = ${:.4f}'.format(sp["mu"], sp["S"] / sp["Np"]))
            if sp["N_BEC"] != 0:
                util.plot_delta_fn(ax, x0 = sp["E_orbs"][0], text = sp["N_BEC"], a_plt = max_DoS[i] * 0.8,
                                   axis = 'y', color = pDoS[0].get_color(), head_width = 0.5, head_length = 0.3)

        ax_DoS.set_xlabel("DoS [arb.]")
        ax_DoS.set_ylabel("E/t")
        ax_DoS.set_title(r'DoS ($T = ${:.2f})'.format(self.T))
        ax_DoS.set_xlim(0, max(max_DoS) * 1.5)
        ax_DoS.legend()
        return ax_DoS
    
    def tabulate_params(self,
                        ax = None,
                        hidden: list[str] = ["DoS", "E_orbs"],
                        str_len: int = 10):
        """Tabulate all the currently available parameters.
        
        "comment" is a dictionary of str. It is currently handled in an ugly way.
            hidden: a list of keys to ignore.
            str_len: length of the string displayed. Long strings are truncated, and short
                     strings are padded with spaces to the left."""
        _, ax_table = util.make_simple_axes(ax, fignum = 2)
        
        displayed_keys = list(self.subregion_list[0].keys())
        for x in hidden: displayed_keys.remove(x)
        all_values_str = [[str(sp[key]).rjust(str_len)[:str_len] for key in displayed_keys]
                          for sp in self.subregion_list]
        plot_table(ax_table, cellText = list(zip(*all_values_str)),
                   rowLabels = displayed_keys, loc = 'center')
        return ax_table

muVT_subregion = namedtuple("muVT_species", ["name", "species", "V", "stat", "DoS", "E_range", "dgn_list", "E_orbs"])
# "name": str                       # name of the subregion
# "species": str                    # name of the species
# "V": int                          # number of orbitals / size of the subregion
# "stat": 1 or -1                   # 1 for fermions, -1 for bosons
    # Probalby won't work well for bosons - I haven't included N_BEC yet
# "DoS": callable like f(E) or None # density of states (Mainly used for plots)
# "E_range": (float, float)         # energies considered in calculation and plots
# "dgn_list": list[tuple]           # A list of tuple(energy: float, num_of_degenerate_orbitals: int)
# "E_orbs": array_like[float]       # (can be None) A complete list of energies of all orbitals considered

class muVT_exp():
    """Grand canonical ensemble (fix (mu, V, T))."""
    def __init__(self, T: float, subregion_list: list[muVT_subregion], mu_dict: dict[float]):
        """Find all the thermal dynamical values and store them as attributes of the object.
        
        Reservoirs are treated on equal footings with systems.
        Np (number of particles) in each species is NOT rounded to integer.
        Args:
            T: temperature of the system.
            subregion_list: list[muVT_subregion] that defines the experiment parameters.
            mu_dict: a dictionary that contains the chemical potentials of each species.
                e.g. {"fermi1": 0.4, "fermi2": 0.05, "bose1": -0.1}
        Attr:
            subregion_list: the E_orbs are generated in __init__ if they are None in the input.
                Otherwise this list is the same as the input subregion_list.
            _input_subregion_list: this is exactly the same as the input.
            results: stores derived quantities as their values, with the subregion names being the keys.
                e.g. {"system": {"Np": 777.8, ...}
                      "reservoir": {"Np": 11111.3, ...}}
            N_dict: total particle number for each species.
            S: total entropy of all the subregions in the experiment.
            E: total energy of all the subregions in the experiment."""
        self.T = T
        self._input_subregion_list = subregion_list
        self.mu_dict = mu_dict
        processed_sr_list = [None for _ in subregion_list]

        # Subregion specific values
        self.results = dict()
        for i, sr in enumerate(subregion_list):
            # E_orbs is usually automatically generated, but can be overridden if the input
            # passes the consistency check
            if sr.E_orbs is None:
                V_dgn = sum([dgn[1] for dgn in sr.dgn_list])
                E_orbs_sr = thmdy.E_orbs_with_deg(sr.DoS, sr.E_range, sr.V - V_dgn, dgn_list = sr.dgn_list)
                sr = muVT_subregion(sr.name, sr.species, sr.V, sr.stat, sr.DoS, sr.E_range, sr.dgn_list, E_orbs_sr)
            if len(sr.E_orbs) != sr.V:
                logging.error("Number of orbits for {} is inconsistent".format(sr.name))
                logging.error("(V = {}, len(E_orbs) = {})".format(sr.V, len(sr.E_orbs)))
                raise(Exception("E_orbs_number_error"))
            else:
                processed_sr_list[i] = sr

            # Initialization
            result_i = dict()
            mu_i = mu_dict[sr.species]
            
            # Calculate thermodynamical values
            result_i["Np"] = thmdy.find_Np(sr.E_orbs, self.T, mu_i, sr.stat)
            result_i["E"] = thmdy.find_E(sr.E_orbs, self.T, mu_i, sr.stat)
            result_i["S"] = thmdy.find_S(sr.E_orbs, self.T, result_i["Np"], sr.stat, mu_i, result_i["E"])
            result_i["nu"] = result_i["Np"] / sr.V
            self.results[sr.name] = result_i
        self.subregion_list = processed_sr_list
        
        # Experiment-level values
        self.N_dict = self.find_N_tot()
        self.S = sum([rst["S"] for rst in self.results.values()])
        self.E = sum([rst["E"] for rst in self.results.values()])

    # Helper thermodynamical functions
    def find_N_tot(self) -> dict:
        """Find the total number of particles for each species.
        
        Returns:
            N_tot: looks like e.g. {"fermi1": 1300.4, "fermi2": 400.3, ...}"""
        N_tot = defaultdict(float)
        for sr in self.subregion_list:
            N_tot[sr.species] += self.results[sr.name]["Np"]
        return N_tot

    ### plot related methods
    def plot_DoSs(self, ax = None, offset_traces: float = 0.1):
        """Visulaization of DoS + filling.
        
        Args:
            offset_traces: offset each DoS trace by offset_traces horizontally."""
        _, ax_DoS = util.make_simple_axes(ax, fignum = 1)

        # Plot DoS and filling
        max_DoS = np.zeros(len(self.subregion_list))
        for i, sr in enumerate(self.subregion_list):
            mu_i = self.mu_dict[sr.species]
            if sr.DoS is None:
                # Consider plotting something similar with histograms
                print("No DoS is given for {}".format(sr.name))
            else:
                # Energies used for plotting is generated separately (less points than E_orbs and don't reflect DoS)
                E_orbs_plot = np.linspace(sr.E_range[0] - 0.5, sr.E_range[1] + 0.5, 1000)
                DoS_plot = sr.DoS(E_orbs_plot)
                max_DoS[i] = max(DoS_plot)
                filling = DoS_plot * util.part_stat(E_orbs_plot, self.T, mu_i, sr.stat)
                
                # plotting
                off_h = 0
                if offset_traces:
                    off_h = offset_traces * i * max_DoS[i]
                    ax_DoS.axvline(off_h, color = 'k', ls = '-', lw = 0.5)
                pDoS = ax_DoS.plot(DoS_plot + off_h, E_orbs_plot, '-', label = sr.name)
                clr = pDoS[0].get_color()
                # if sr.species not in colors_used.keys():
                #     colors_used[sr.species] = pDoS[0].get_color()
                ax_DoS.fill_betweenx(E_orbs_plot, x1 = filling + off_h, x2 = off_h
                                    , ls = '--', alpha = 0.3)
                ax_DoS.axhline(mu_i, color = clr, ls = '--'
                            , label = r'$\mu = ${:.3f}, $s = ${:.4f}'.format(mu_i, self.results[sr.name]["S"] / self.results[sr.name]["Np"]))
                print("xlim = {}, ylim = {}".format(ax_DoS.get_xlim(), ax_DoS.get_ylim()))
                for dgn in sr.dgn_list:
                    nu_dgn = util.part_stat(dgn[0], self.T, mu_i, sr.stat)
                    N_dgn = dgn[1] * nu_dgn
                    util.plot_delta_fn(ax_DoS, dgn[0], 2, text = r"$N_{orbs} = $" + "{:.2f}".format(dgn[1]), axis = "y", color = clr)
                    util.plot_delta_fn(ax_DoS, dgn[0], 2 * nu_dgn, text = r"$N_{fill} = $" + "{:.2f}".format(N_dgn), axis = "y", color = clr, alpha = 0.3, text_height = -1)

        ax_DoS.set_xlabel("DoS [arb.]")
        ax_DoS.set_ylabel("E/t")
        ax_DoS.set_title(r'DoS ($T = ${:.4f})'.format(self.T))
        # ax_DoS.set_xlim(0, max(max_DoS) * 1.5)
        ax_DoS.legend()
        return ax_DoS
    
    def tabulate_params(self,
                        ax = None,
                        hidden: list[str] = ["DoS", "E_orbs"],
                        str_len: int = 10):
        """Tabulate all the currently available parameters.
        
        Args:
            hidden: a list of keys to ignore.
            str_len: length of the string displayed. Long strings are truncated, and short
                     strings are padded with spaces to the left."""
        _, ax_table = util.make_simple_axes(ax, fignum = 2)
        
        all_init_fields = list(muVT_subregion._fields)
        all_derv_fields = list(list(self.results.values())[0].keys()) # Gross!
        for fields in (all_init_fields, all_derv_fields):
            for field in fields:
                if field in hidden:
                    fields.remove(field)
        displayed_keys = all_init_fields + all_derv_fields
        all_values_str = [
            [str(getattr(sr, key)).rjust(str_len)[:str_len] for key in all_init_fields] +
            [str(self.results[sr.name][key]).rjust(str_len)[:str_len] for key in all_derv_fields]
                        for sr in self.subregion_list]
        
        plot_table(ax_table, cellText = list(zip(*all_values_str)),
                   rowLabels = displayed_keys, loc = 'center')
        return ax_table
    
    def plot_E_orb(self, fbw = 0.2):
        """Plots all the orbitals used in the calculation.
        
        Good for debugging purposes.
        Args:
            fbw: band width of what is considered a flat band state."""
        for i, sr in enumerate(self.subregion_list):
            f = plt.figure(i + 3, figsize = (15, 15))
            ax_scat = f.add_subplot(121)
            ax_hist = f.add_subplot(122)

            EOI = (sr.E_range[1] - fbw, sr.E_range[1] + fbw)
            num_in_EOI = sum(np.logical_and(sr.E_orbs > EOI[0], sr.E_orbs < EOI[1]))
            random_offsets = 0.3 * np.random.default_rng().standard_normal(len(sr.E_orbs))
            ax_scat.scatter(random_offsets, sr.E_orbs, s = 1, marker = ".")
            ax_scat.fill_between(ax_scat.get_xlim(), (EOI[0], EOI[0]), (EOI[1], EOI[1]), alpha = 0.3)

            E_bins = np.linspace(sr.E_range[0], sr.E_range[1], 500)
            N, bins, patches = ax_hist.hist(sr.E_orbs, bins = E_bins)
            f.suptitle("Ratio of orbitals in ({}, {}): {} (num = {})".format(EOI[0], EOI[1], num_in_EOI / len(sr.E_orbs), num_in_EOI))
        return ax_scat, ax_hist