import simple_fermi_bose as sfb
import fermi_with_reservoir as fwr

def main():
    module = sfb
    kwargs = {"calculation_mode": "isentropic"}
    module.main(**kwargs)

if __name__ == "__main__":
    main()