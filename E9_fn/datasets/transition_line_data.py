# Recommended import call: import E9_fn.datasets.transition_line_data as TLData

# All the data is from NIST database (line data for Rb I)
# format: (isotope)_gs[n(term symbol)]_es[n(term symbol)]; 9/2 -> 9o2 etc
# lambda: [m] wavelength of the transition; USE VACUUM WAVELENGTH! (I use Ritz wavelength if available)
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
Rb_5S1o2_5P1o2 = {'lambda': 794.9789e-9, 'A_ki': 3.614e+07, 'f_ik': 3.424e-01, 'acc': 'AAA',
                  'gs': '5S1o2', 'es': '5P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_6P1o2 = {'lambda': 421.6726e-9, 'A_ki': 1.50e+06, 'f_ik': 4.00e-03, 'acc': 'B',
                  'gs': '5S1o2', 'es': '6P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_7P1o2 = {'lambda': 359.25967e-9, 'A_ki': 2.89e+05, 'f_ik': 5.60e-04, 'acc': 'B',
                  'gs': '5S1o2', 'es': '7P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_8P1o2 = {'lambda': 335.17748e-9, 'A_ki': 8.91e+04, 'f_ik': 1.50e-04, 'acc': 'C+',
                  'gs': '5S1o2', 'es': '8P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_9P1o2 = {'lambda': 323.00879e-9, 'A_ki': 3.84e+04, 'f_ik': 6.00e-05, 'acc': 'C+',
                  'gs': '5S1o2', 'es': '9P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_10P1o2 = {'lambda': 315.91734e-9, 'A_ki': 2.01e+04, 'f_ik': 3.00e-05, 'acc': 'B',
                  'gs': '5S1o2', 'es': '10P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}
Rb_5S1o2_11P1o2 = {'lambda': 311.39503e-9, 'A_ki': 1.27e+04, 'f_ik': 1.85e-05, 'acc': 'B',
                  'gs': '5S1o2', 'es': '11P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'Rb'}

# Rb, transitions to P3/2 lines (including D2 line)
Rb_5S1o2_5P3o2 = {'lambda': 780.2415e-9, 'A_ki': 3.812e+07, 'f_ik': 6.958e-01, 'acc': 'AAA',
                  'gs': '5S1o2', 'es': '5P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_6P3o2 = {'lambda': 420.29891e-9, 'A_ki': 1.77e+06, 'f_ik': 9.37e-03, 'acc': 'B+',
                  'gs': '5S1o2', 'es': '6P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_7P3o2 = {'lambda': 358.80734e-9, 'A_ki': 3.96e+05, 'f_ik': 1.53e-03, 'acc': 'B+',
                  'gs': '5S1o2', 'es': '7P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_8P3o2 = {'lambda': 334.96585e-9, 'A_ki': 1.37e+05, 'f_ik': 4.60e-04, 'acc': 'B',
                  'gs': '5S1o2', 'es': '8P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_9P3o2 = {'lambda': 322.89114e-9, 'A_ki': 6.40e+04, 'f_ik': 2.00e-04, 'acc': 'C+',
                  'gs': '5S1o2', 'es': '9P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_10P3o2 = {'lambda': 315.84440e-9, 'A_ki': 3.38e+04, 'f_ik': 1.01e-04, 'acc': 'B',
                  'gs': '5S1o2', 'es': '10P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}
Rb_5S1o2_11P3o2 = {'lambda': 311.34685e-9, 'A_ki': 2.51e+04, 'f_ik': 7.30e-05, 'acc': 'B',
                  'gs': '5S1o2', 'es': '11P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}

# Rb E1 forbidden lines
Rb_5S1o2_4D5o2 = {'lambda': 516.6569e-9, 'A_ki': None, 'f_ik': None, 'acc': None,
                  'gs': '5S1o2', 'es': '4D5o2', 'Jg': 1/2, 'Je': 5/2, 'isotope': 'Rb'}
Rb_5S1o2_4D3o2 = {'lambda': 516.6450e-9, 'A_ki': None, 'f_ik': None, 'acc': None,
                  'gs': '5S1o2', 'es': '4D3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'Rb'}

# K, transitions from 4S1/2 to nP1/2 lines (including D1 line)
K_4S1o2_4P1o2 = {'lambda': 770.10835e-9, 'A_ki': 3.734e+07, 'f_ik': 3.3201e-01, 'acc': 'AAA',
                  'gs': '4S1o2', 'es': '4P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_5P1o2 = {'lambda': 404.83565e-9, 'A_ki': 1.07e+06, 'f_ik': 2.63e-03, 'acc': 'B+',
                  'gs': '4S1o2', 'es': '5P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_6P1o2 = {'lambda': 344.8363e-9, 'A_ki': 1.45e+05, 'f_ik': 2.58e-04, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '6P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_7P1o2 = {'lambda': 321.8549e-9, 'A_ki': 4.00e+04, 'f_ik': 6.22e-05, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '7P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_8P1o2 = {'lambda': 310.2946e-9, 'A_ki': 1.22e+04, 'f_ik': 1.76e-05, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '8P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_9P1o2 = {'lambda': 303.58040e-9, 'A_ki': 4.78e+03, 'f_ik': 6.60e-06, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '9P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4S1o2_10P1o2 = {'lambda': 299.30952e-9, 'A_ki': 2.12e+03, 'f_ik': 2.85e-06, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '10P1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}

# K, transitions from 4s1/2 to nP3/2 lines (including D2 line)
K_4S1o2_4P3o2 = {'lambda': 766.70089e-9, 'A_ki': 3.779e+07, 'f_ik': 6.661e-01, 'acc': 'AAA',
                  'gs': '4S1o2', 'es': '4P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_5P3o2 = {'lambda': 404.52847e-9, 'A_ki': 1.15e+06, 'f_ik': 5.67e-03, 'acc': 'A',
                  'gs': '4S1o2', 'es': '5P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_6P3o2 = {'lambda': 344.7359e-9, 'A_ki': 1.65e+05, 'f_ik': 5.89e-04, 'acc': 'B+',
                  'gs': '4S1o2', 'es': '6P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_7P3o2 = {'lambda': 321.8083e-9, 'A_ki': 5.01e+04, 'f_ik': 1.55e-04, 'acc': 'B+',
                  'gs': '4S1o2', 'es': '7P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_8P3o2 = {'lambda': 310.2689e-9, 'A_ki': 1.62e+04, 'f_ik': 4.66e-05, 'acc': 'B+',
                  'gs': '4S1o2', 'es': '8P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_9P3o2 = {'lambda': 303.56452e-9, 'A_ki': 6.74e+03, 'f_ik': 1.86e-05, 'acc': 'B',
                  'gs': '4S1o2', 'es': '9P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_10P3o2 = {'lambda': 299.29905e-9, 'A_ki': 3.16e+03, 'f_ik': 8.50e-06, 'acc': 'C+',
                  'gs': '4S1o2', 'es': '10P3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}

# K, other 4s1/2 E1 forbidden lines
K_4S1o2_3D5o2 = {'lambda': 464.36724e-9, 'A_ki': 1.54e+02, 'f_ik': None, 'acc': 'AAA',
                  'gs': '4S1o2', 'es': '3D5o2', 'Jg': 1/2, 'Je': 5/2, 'isotope': 'K'}
K_4S1o2_3D3o2 = {'lambda': 464.31748e-9, 'A_ki': 1.54e+02, 'f_ik': None, 'acc': 'AAA',
                  'gs': '4S1o2', 'es': '3D5o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4S1o2_4D5o2 = {'lambda': 365.00244e-9, 'A_ki': None, 'f_ik': None, 'acc': None,
                  'gs': '4S1o2', 'es': '4D5o2', 'Jg': 1/2, 'Je': 5/2, 'isotope': 'K'}
K_4S1o2_4D3o2 = {'lambda': 364.98819e-9, 'A_ki': None, 'f_ik': None, 'acc': None,
                  'gs': '4S1o2', 'es': '4D3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}

# K, 4P1/2 to nS1/2 lines (not including D1 line)
K_4P1o2_5S1o2 = {'lambda': 1243.5699e-9, 'A_ki': 7.951e+06, 'f_ik': 1.843e-01, 'acc': 'AA',
                  'gs': '4P1o2', 'es': '5S1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4P1o2_6S1o2 = {'lambda': 691.29881e-9, 'A_ki': 2.500e+06, 'f_ik': 1.791e-02, 'acc': 'AA',
                  'gs': '4P1o2', 'es': '6S1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4P1o2_7S1o2 = {'lambda': 578.40035e-9, 'A_ki': 1.186e+06, 'f_ik': 5.949e-03, 'acc': 'AA',
                  'gs': '4P1o2', 'es': '7S1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}
K_4P1o2_8S1o2 = {'lambda': 532.47595e-9, 'A_ki': 6.62e+05, 'f_ik': 2.812e-03, 'acc': 'A+',
                  'gs': '4P1o2', 'es': '8S1o2', 'Jg': 1/2, 'Je': 1/2, 'isotope': 'K'}

# K, 4P1/2 to nDJ lines
K_4P1o2_3D3o2 = {'lambda': 1169.3442e-9, 'A_ki': 2.017e+07, 'f_ik': 8.269e-01, 'acc': 'AA',
                  'gs': '4P1o2', 'es': '3D3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4P1o2_4D3o2 = {'lambda': 693.81995e-9, 'A_ki': 1.9e+04, 'f_ik': 2.7e-04, 'acc': 'C',
                  'gs': '4P1o2', 'es': '4D3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4P1o2_5D3o2 = {'lambda': 581.37636e-9, 'A_ki': 2.86e+05, 'f_ik': 2.90e-03, 'acc': 'A',
                  'gs': '4P1o2', 'es': '5D3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}
K_4P1o2_6D3o2 = {'lambda': 534.44554e-9, 'A_ki': 3.86e+05, 'f_ik': 3.30e-03, 'acc': 'A',
                  'gs': '4P1o2', 'es': '6D3o2', 'Jg': 1/2, 'Je': 3/2, 'isotope': 'K'}

# K, 4P3/2 to nS1/2 lines (not including D2 line)
K_4P3o2_5S1o2 = {'lambda': 1252.5591e-9, 'A_ki': 1.582e+07, 'f_ik': 1.861e-01, 'acc': 'AA',
                  'gs': '4P3o2', 'es': '5S1o2', 'Jg': 3/2, 'Je': 1/2, 'isotope': 'K'}
K_4P3o2_6S1o2 = {'lambda': 694.06780e-9, 'A_ki': 4.956e+06, 'f_ik': 1.790e-02, 'acc': 'AA',
                  'gs': '4P3o2', 'es': '6S1o2', 'Jg': 3/2, 'Je': 1/2, 'isotope': 'K'}
K_4P3o2_7S1o2 = {'lambda': 580.33749e-9, 'A_ki': 2.348e+06, 'f_ik': 5.927e-03, 'acc': 'AA',
                  'gs': '4P3o2', 'es': '7S1o2', 'Jg': 3/2, 'Je': 1/2, 'isotope': 'K'}
K_4P3o2_8S1o2 = {'lambda': 534.11726e-9, 'A_ki': 1.311e+06, 'f_ik': 2.803e-03, 'acc': 'AA',
                  'gs': '4P3o2', 'es': '8S1o2', 'Jg': 3/2, 'Je': 1/2, 'isotope': 'K'}

# K, 4P3/2 to nDJ lines
K_4P3o2_3D5o2 = {'lambda': 1177.6089e-9, 'A_ki': 2.383e+07, 'f_ik': 7.430e-01, 'acc': 'AA',
                  'gs': '4P3o2', 'es': '3D5o2', 'Jg': 3/2, 'Je': 5/2, 'isotope': 'K'}
K_4P3o2_3D3o2 = {'lambda': 1177.2889e-9, 'A_ki': 3.974e+06, 'f_ik': 8.258e-02, 'acc': 'AA',
                  'gs': '4P3o2', 'es': '3D3o2', 'Jg': 3/2, 'Je': 3/2, 'isotope': 'K'}
K_4P3o2_4D5o2 = {'lambda': 696.66113e-9, 'A_ki': 1.37e+04, 'f_ik': 1.49e-04, 'acc': 'D+',
                  'gs': '4P3o2', 'es': '4D5o2', 'Jg': 3/2, 'Je': 5/2, 'isotope': 'K'}
K_4P3o2_4D3o2 = {'lambda': 696.60921e-9, 'A_ki': 2.4e+03, 'f_ik': 1.7e-05, 'acc': 'D+',
                  'gs': '4P3o2', 'es': '4D3o2', 'Jg': 3/2, 'Je': 3/2, 'isotope': 'K'}
K_4P3o2_5D5o2 = {'lambda': 583.35066e-9, 'A_ki': 3.71e+05, 'f_ik': 2.84e-03, 'acc': 'A',
                  'gs': '4P3o2', 'es': '5D5o2', 'Jg': 3/2, 'Je': 5/2, 'isotope': 'K'}
K_4P3o2_5D3o2 = {'lambda': 583.33352e-9, 'A_ki': 6.13e+04, 'f_ik': 3.13e-04, 'acc': 'A',
                  'gs': '4P3o2', 'es': '5D3o2', 'Jg': 3/2, 'Je': 3/2, 'isotope': 'K'}
K_4P3o2_6D5o2 = {'lambda': 536.10666e-9, 'A_ki': 4.86e+05, 'f_ik': 3.14e-03, 'acc': 'A',
                  'gs': '4P3o2', 'es': '6D5o2', 'Jg': 3/2, 'Je': 5/2, 'isotope': 'K'}
K_4P3o2_6D3o2 = {'lambda': 536.09903e-9, 'A_ki': 8.10e+04, 'f_ik': 3.49e-04, 'acc': 'A',
                  'gs': '4P3o2', 'es': '6D3o2', 'Jg': 3/2, 'Je': 3/2, 'isotope': 'K'}

#%% All lines
# Rb
Rb_5S1o2_nP1o2_lines = [Rb_5S1o2_5P1o2, Rb_5S1o2_6P1o2, Rb_5S1o2_7P1o2, Rb_5S1o2_8P1o2, Rb_5S1o2_9P1o2,
                        Rb_5S1o2_10P1o2, Rb_5S1o2_11P1o2]
Rb_5S1o2_nP3o2_lines = [Rb_5S1o2_5P3o2, Rb_5S1o2_6P3o2, Rb_5S1o2_7P3o2, Rb_5S1o2_8P3o2, Rb_5S1o2_9P3o2,
                        Rb_5S1o2_10P3o2, Rb_5S1o2_11P3o2]
Rb_5S1o2_other_lines = [Rb_5S1o2_4D5o2, Rb_5S1o2_4D3o2]
Rb_5S1o2_lines = Rb_5S1o2_nP1o2_lines + Rb_5S1o2_nP3o2_lines + Rb_5S1o2_other_lines
Rb_D12_doublet = [Rb_5S1o2_5P1o2, Rb_5S1o2_5P3o2]

# K
K_4S1o2_nP1o2_lines = [K_4S1o2_4P1o2, K_4S1o2_5P1o2, K_4S1o2_6P1o2, K_4S1o2_7P1o2, K_4S1o2_8P1o2,
                       K_4S1o2_9P1o2]
K_4S1o2_nP3o2_lines = [K_4S1o2_4P3o2, K_4S1o2_5P3o2, K_4S1o2_6P3o2, K_4S1o2_7P3o2, K_4S1o2_8P3o2,
                       K_4S1o2_9P3o2]
K_4S1o2_other_lines = [K_4S1o2_3D5o2, K_4S1o2_3D3o2, K_4S1o2_4D5o2, K_4S1o2_4D3o2]
K_4S1o2_lines = K_4S1o2_nP1o2_lines + K_4S1o2_nP3o2_lines + K_4S1o2_other_lines

K_4P3o2_nS1o2_lines = [K_4S1o2_4P3o2, K_4P3o2_5S1o2, K_4P3o2_6S1o2, K_4P3o2_7S1o2, K_4P3o2_8S1o2] # including D2 line
K_4P3o2_nDJ_lines = [K_4P3o2_3D5o2, K_4P3o2_3D3o2, K_4P3o2_4D5o2, K_4P3o2_4D3o2, K_4P3o2_5D5o2, K_4P3o2_5D3o2,
                     K_4P3o2_6D5o2, K_4P3o2_6D3o2]
K_4P3o2_lines = K_4P3o2_nS1o2_lines + K_4P3o2_nDJ_lines

K_4P1o2_nS1o2_lines = [K_4S1o2_4P1o2, K_4P1o2_5S1o2, K_4P1o2_6S1o2, K_4P1o2_7S1o2, K_4P1o2_8S1o2] # including D2 line
K_4P1o2_nDJ_lines = [K_4P1o2_3D3o2, K_4P1o2_4D3o2, K_4P1o2_5D3o2, K_4P1o2_6D3o2]
K_4P1o2_lines = K_4P1o2_nS1o2_lines + K_4P1o2_nDJ_lines