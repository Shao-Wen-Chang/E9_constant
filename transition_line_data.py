# All the data is from NIST database (line data for Rb I)
# format: (isotope)_gs[n(term symbol)]_es[n(term symbol)]; 9/2 -> 9o2 etc
# lambda: [m] wavelength of the transition; I use Ritz wavelength if available
# f_ik: [dimless] oscillator strength
# acc: "Accuracy" of transition strength
# gs: ground state; I don't need to deal with crazy states yet so they are just written as nLJ for now
# es: excited state
# Jg: J of ground state (J)
# Je: J of excited state (J')
# isotope: atomic species, including the number of nucleons if applicable

#%% Lines to be added to all lines
### Typically list from low to high energy (long to short wavelengths)
# For Rb there are data of oscillator strengths up to 20p, but they are very weak
# Rb, transitions to P1/2 lines (including D1 line)
Rb_5S1o2_5P1o2 = {'lambda': 794.7603e-9, 'A_ki': 3.614e+07, 'f_ik': 3.424e-01, 'acc': 'AAA',
                  'gs': '5S1o2', 'es': '5P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_6P1o2 = {'lambda': 421.5539e-9, 'A_ki': 1.50e+06, 'f_ik': 4.00e-03, 'acc': 'B',
                  'gs': '5S1o2', 'es': '6P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_7P1o2 = {'lambda': 359.15717e-9, 'A_ki': 2.89e+05, 'f_ik': 5.60e-04, 'acc': 'B',
                  'gs': '5S1o2', 'es': '7P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_8P1o2 = {'lambda': 335.08116e-9, 'A_ki': 8.91e+04, 'f_ik': 1.50e-04, 'acc': 'C+',
                  'gs': '5S1o2', 'es': '8P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_9P1o2 = {'lambda': 322.91557e-9, 'A_ki': 3.84e+04, 'f_ik': 6.00e-05, 'acc': 'C+',
                  'gs': '5S1o2', 'es': '9P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_10P1o2 = {'lambda': 315.82591e-9, 'A_ki': 2.01e+04, 'f_ik': 3.00e-05, 'acc': 'B',
                  'gs': '5S1o2', 'es': '10P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}

# Rb, transitions to P3/2 lines (including D2 line)
Rb_5S1o2_5P3o2 = {'lambda': 780.0268e-9, 'A_ki': 3.812e+07, 'f_ik': 6.958e-01, 'acc': 'AAA',
                  'gs': '5S1o2', 'es': '5P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_6P3o2 = {'lambda': 420.18053e-9, 'A_ki': 1.77e+06, 'f_ik': 9.37e-03, 'acc': 'B+',
                  'gs': '5S1o2', 'es': '6P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_7P3o2 = {'lambda': 358.70496e-9, 'A_ki': 3.96e+05, 'f_ik': 1.53e-03, 'acc': 'B+',
                  'gs': '5S1o2', 'es': '7P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_8P3o2 = {'lambda': 334.86958e-9, 'A_ki': 1.37e+05, 'f_ik': 4.60e-04, 'acc': 'B',
                  'gs': '5S1o2', 'es': '8P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_9P3o2 = {'lambda': 322.79795e-9, 'A_ki': 6.40e+04, 'f_ik': 2.00e-04, 'acc': 'C+',
                  'gs': '5S1o2', 'es': '9P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_10P3o2 = {'lambda': 315.75298e-9, 'A_ki': 3.38e+04, 'f_ik': 1.01e-04, 'acc': 'B',
                  'gs': '5S1o2', 'es': '10P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}

# Rb E1 forbidden lines
Rb_5S1o2_4D5o2 = {'lambda': 516.5131e-9, 'A_ki': None, 'f_ik': None, 'acc': None,
                  'gs': '5S1o2', 'es': '4D5o2', 'Jg': 1/2, 'Je': 5/2, 'isotope': 'Rb'}
Rb_5S1o2_4D3o2 = {'lambda': 516.5012e-9, 'A_ki': None, 'f_ik': None, 'acc': None,
                  'gs': '5S1o2', 'es': '4D3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}

# K, transitions to P1/2 lines (including D1 line)
K_4S1o2_4P1o2 = {'lambda': 769.89646e-9, 'A_ki': 3.734e+07, 'f_ik': 3.3201e-01, 'acc': 'AAA',
                  'gs': '4S1o2', 'es': '4P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_5P1o2 = {'lambda': 404.72132e-9, 'A_ki': 1.07e+06, 'f_ik': 2.63e-03, 'acc': 'B+',
                  'gs': '4S1o2', 'es': '5P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_6P1o2 = {'lambda': 344.7375e-9, 'A_ki': 1.45e+05, 'f_ik': 2.58e-04, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '6P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_7P1o2 = {'lambda': 321.7620e-9, 'A_ki': 4.00e+04, 'f_ik': 6.22e-05, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '7P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_8P1o2 = {'lambda': 310.2046e-9, 'A_ki': 1.22e+04, 'f_ik': 1.76e-05, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '8P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_9P1o2 = {'lambda': 303.49207e-9, 'A_ki': 4.78e+03, 'f_ik': 6.60e-06, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '9P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}

# K, transitions to P3/2 lines (including D2 line)
K_4S1o2_4P3o2 = {'lambda': 766.48991e-9, 'A_ki': 3.779e+07, 'f_ik': 6.661e-01, 'acc': 'AAA',
                  'gs': '4S1o2', 'es': '4P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_5P3o2 = {'lambda': 404.41422e-9, 'A_ki': 1.15e+06, 'f_ik': 5.67e-03, 'acc': 'A',
                  'gs': '4S1o2', 'es': '5P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_6P3o2 = {'lambda': 344.6372e-9, 'A_ki': 1.65e+05, 'f_ik': 5.89e-04, 'acc': 'B+',
                  'gs': '4S1o2', 'es': '6P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_7P3o2 = {'lambda': 321.7154e-9, 'A_ki': 5.01e+04, 'f_ik': 1.55e-04, 'acc': 'B+',
                  'gs': '4S1o2', 'es': '7P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_8P3o2 = {'lambda': 310.1789e-9, 'A_ki': 1.62e+04, 'f_ik': 4.66e-05, 'acc': 'B+',
                  'gs': '4S1o2', 'es': '8P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_9P3o2 = {'lambda': 303.47619e-9, 'A_ki': 6.74e+03, 'f_ik': 1.86e-05, 'acc': 'B',
                  'gs': '4S1o2', 'es': '9P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}

# K E1 forbidden lines
K_4S1o2_3D5o2 = {'lambda': 464.23725e-9, 'A_ki': 1.54e+02, 'f_ik': None, 'acc': 'AAA',
                  'gs': '4S1o2', 'es': '3D5o2', 'Jg': 1/2, 'Je': 5/2, 'isotope': 'K'}
K_4S1o2_3D3o2 = {'lambda': 464.18750e-9, 'A_ki': 1.54e+02, 'f_ik': None, 'acc': 'AAA',
                  'gs': '4S1o2', 'es': '3D5o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_4D5o2 = {'lambda': 364.89847e-9, 'A_ki': None, 'f_ik': None, 'acc': None,
                  'gs': '4S1o2', 'es': '4D5o2', 'Jg': 1/2, 'Je': 5/2, 'isotope': 'K'}
K_4S1o2_4D3o2 = {'lambda': 364.88422e-9, 'A_ki': None, 'f_ik': None, 'acc': None,
                  'gs': '4S1o2', 'es': '4D3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}

#%% All lines
Rb_Je1o2_lines = [Rb_5S1o2_5P1o2, Rb_5S1o2_6P1o2, Rb_5S1o2_7P1o2, Rb_5S1o2_8P1o2, Rb_5S1o2_9P1o2,
                  Rb_5S1o2_10P1o2]
Rb_Je3o2_lines = [Rb_5S1o2_5P3o2, Rb_5S1o2_6P3o2, Rb_5S1o2_7P3o2, Rb_5S1o2_8P3o2, Rb_5S1o2_9P3o2,
                  Rb_5S1o2_10P3o2]
Rb_other_lines = [Rb_5S1o2_4D5o2, Rb_5S1o2_4D3o2]
Rb_gs_lines = Rb_Je1o2_lines + Rb_Je3o2_lines + Rb_other_lines

K_Je1o2_lines = [K_4S1o2_4P1o2, K_4S1o2_5P1o2, K_4S1o2_6P1o2, K_4S1o2_7P1o2, K_4S1o2_8P1o2,
                 K_4S1o2_9P1o2]
K_Je3o2_lines = [K_4S1o2_4P3o2, K_4S1o2_5P3o2, K_4S1o2_6P3o2, K_4S1o2_7P3o2, K_4S1o2_8P3o2,
                 K_4S1o2_9P3o2]
K_other_lines = [K_4S1o2_3D5o2, K_4S1o2_3D3o2, K_4S1o2_4D5o2, K_4S1o2_4D3o2]
K_gs_lines = K_Je1o2_lines + K_Je3o2_lines + K_other_lines