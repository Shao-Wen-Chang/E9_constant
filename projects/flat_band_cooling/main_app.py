import logging
import time
import sys

import simple_fermi_bose as sfb
import simple_fermi_fermi as sff
import fermi_with_reservoir as fwr
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
# Main "interface" for running simulations
# Remember that all the experiments considered here does not include interactions,
# so they don't allow thermalization as is. Except for the case of spinless fermions,
# one can imagine turning on a small interaction and wait for thermalization, and
# then slowly ramp interaction to zero. Then the results should apply reasonably well.

def main():
    util.set_custom_plot_style(True, overwrite = {"font.size": 11})

    # User inputs
    logpath = '' # '' if not logging to a file
    loglevel = logging.INFO
    module = sff
    kwargs = {"calculation_mode": "isentropic"
              } # any arguments that can be passed to the main() of the module
    
    # Configuring logger (reset and reconfigured at each run)
    start_time = time.time()
    logroot = logging.getLogger()
    list(map(logroot.removeHandler, logroot.handlers))
    list(map(logroot.removeFilter, logroot.filters))
    logging.basicConfig(filename = logpath, level = loglevel)

    # Run module main()
    logging.info("main started")
    module.main(**kwargs)
    logging.info("main exited successfully, time = {:.2f}s".format(time.time() - start_time))

if __name__ == "__main__":
    main()