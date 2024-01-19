import logging
import time

import simple_fermi_bose as sfb
import fermi_with_reservoir as fwr

def main():
    # User inputs
    logpath = '' # '' if not logging to a file
    loglevel = logging.INFO
    module = sfb
    kwargs = {"calculation_mode": "isentropic"
              }
    
    # Configuring logger (reset and reconfigured at each run)
    start_time = time.time()
    logroot = logging.getLogger()
    list(map(logroot.removeHandler, logroot.handlers))
    list(map(logroot.removeFilter, logroot.filters))
    logging.basicConfig(filename = logpath, level = loglevel)
    logging.info("main started")

    # Run module main()
    module.main(**kwargs)
    logging.info("main exited successfully, time = {:.2f}s".format(time.time() - start_time))

if __name__ == "__main__":
    main()