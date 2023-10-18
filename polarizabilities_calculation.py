from E9_constants import *
from E9_numbers import *
from transition_line_data import *
import numpy as np
import matplotlib.pyplot as plt
import util

# Plot parameters are in their own sections below
pol_SI2au = 1/1.64877727436e-41                                     # Conversion factor from SI to a.u. for polarizability
wavelengths = np.arange(400, 1600, 0.001) * 1e-9
beam_waist = 50e-6                                                  # the RADIUS of the beam at 1/e^2 intensity
peak_E2 = 2 * I_from_power(1, beam_waist) / c_light / epsilon_0     # |E|**2, assuming a gaussian beam and 1W of optical power
alpha2kHz = peak_E2 / 4 / pol_SI2au / hnobar / 1000                 # Convert from alpha (in a.u.) to trap depth, in kHz
alpha2uK = peak_E2 / 4 / pol_SI2au / k_B * 1e6                      # Convert from alpha (in a.u.) to trap depth, in uK

# What lines to use
K_4S1o2_LOI = K_4S1o2_lines                                         # Use K_4S1o2_lines to include all lines
K_4P1o2_LOI = K_4P1o2_lines                                         # Use K_4P1o2_lines to include all lines
K_4P3o2_LOI = K_4P3o2_lines                                         # Use K_4P3o2_lines to include all lines
Rb_5S1o2_LOI = Rb_D12_doublett                                      # Use Rb_5S1o2_lines to include all lines

#%% functions
def alpha_pol(K, lamb_in, line_list, state, q = None, F = None, I = None):
    '''[A2·s4·kg−1] Returns the (rank-K) polarizability given a list of lines, ignoring scattering.
    
    See [Axner04], esp. for terminologies in Appendix A. You should understand what I mean by: the result applies for
        - a "hf state" to a "hf state," when q, F and I are all given
        - a "hf state" to all E1-allowed "hf states" summed together, when F and I are given
        - a "fs state" to a "fs state," when q is given
        - a "fs state" to all E1-allowed "fs states" summed together, when q, F and I are not given
    Expressions for alphas are taken from [Le Kien13]. (I ignore the imaginary \gamma terms in the expression in that paper.)
        K: [dimless] K = 0/1/2, for scalar/vector/tensor light shifts, respectively
        lamb_in: [nm] wavelength of incident light
        line_list: a list of LINES considered. They should have the same gs.
        state: a string denoting what FINE STRUCTURE (LEVEL) is of interest.
        q: polarization of the photon, such that m' = m + q.
        F: the hyperfine level of the state; contributes to a constant factor. This is only true if one
           can ignore the coupling between different hyperfine state, i.e. when the Stark shift is small
           compared to hyperfine splitting. (In particular, this can be violated when one works with very
           very strong lattices used during microscope imaging.)
        I: required for hyperfine states calculations.'''
    alpha = np.zeros_like(lamb_in)
    wl = 2 * pi * (c_light / lamb_in)
    for line in line_list:
        iso, gs, es = line['isotope'], line['gs'], line['es']
        line_name = iso + '_' + gs + '_' + es
        
        # Sorting out the relevant lines and grabbing line data
        lamb, f_ik = line['lambda'], line['f_ik'] # numbers used in calculation
        if f_ik is None:
            # No oscillator strength (although Einstein A-coefficient might be nonzero)
            print(line_name + ' transition does not have f_ik data (not E1 allowed?)')
            continue
        
        if line['gs'] == state:
            J1, J2 = line['Jg'], line['Je']
            wa = 2 * pi * (c_light / lamb) # angular frequencie of atomic transitions
        elif line['es'] == state:
            J1, J2 = line['Je'], line['Jg']
            wa = - 2 * pi * (c_light / lamb) # (negaive if the state of concern is the excited state)
        else:
            print(line_name + ' transition has no effect and is ignored')
            continue
        
        # Actual calculation
        # This is actually just S_ik in S.I.
        mat_ele_sqr = f_ik * (2 * line['Jg'] + 1) * (3 * hbar * e_ele**2) / (2 * m_e * abs(wa)) # reduced matrix element squared
        if F is None:
            alpha += (-1)**(K + J1 + J2 + 1) * np.sqrt(2*K + 1) * float(wigner_6j(1, K, 1, J1, J2, J1)) * \
                     mat_ele_sqr * (1 / (wa - wl) + (-1)**K / (wa + wl)) / hbar
        else:
            F2min, F2max = max(abs(J2 - I), F - 1), min(abs(J2 + I), F + 1)
            print(line_name + ': F = {}, F\' = {} ~ {}'.format(F, F2min, F2max))
            for F2 in np.arange(F2min, F2max + 1):
                alpha += (-1)**(K + F + F2 + 1) * (2*F + 1) * (2*F2 + 1) * np.sqrt(2*K + 1) * mat_ele_sqr * \
                         float(wigner_6j(1, K, 1, F, F2, F)) * float(wigner_6j(F, 1, F2, J2, I, J1))**2 * \
                         (1 / (wa - wl) + (-1)**K / (wa + wl)) / hbar
    
    if F is None:
        G = J1
    else:
        G = F
        
    if K == 0:
        prefactor = 1 / np.sqrt(3 * (2 * G + 1))
    elif K == 1:
        prefactor = - np.sqrt(2 * G / ((G + 1) * (2 * G + 1)))
    elif K == 2:
        prefactor = - np.sqrt(2 * G * (2 * G - 1) / (3 * (G + 1) * (2 * G + 1) * (2 * G + 3)))

    return prefactor * alpha

def C_av2B(hfs, ul = np.array([1, 0, 0])):
    '''Calculate the conversion factor for effective B field (at E = 1 V/m).
    
    See [Le Kien13] eqn.(21). This can be treated as a magnetic field in almost every sense.
    B_eff = av2Beff * av * |E|**2
    Make sure that the quantization axis used here is consistent with the calculation in which this function is used.
    hfs: a HyperfineState object to get various quantities from
    ul: polarization of light in spherical basis, (A_{-1}, A_0, A_1) = (\sigma_-, \pi, \sigma_+).'''
    u = util.Normalize(ul)
    pol_fac = abs(ul[0])**2 - abs(ul[2])**2 # this is the i(u* x u) term with all the algebra performed
    return pol_fac / (8 * mu_B * hfs.gF * hfs.F)

def mark_important_lines(ax, lines, f_min, l_alpha = None, init_text_height = 0):
    '''Label the important transitions on the plot.
    
    ax: an axes object where the lines are added to
    lines: a list of transition lines
    f_min: [dimless] minimum oscillator strength for a line to be considered important
    l_alpha: a line object to get color from
    init_text_height: related to how the text labels for transitions are cycled through in height'''
    y_min = ax.get_ylim()[0]
    text_ypos = init_text_height
    if l_alpha is None:
        vcolor = '#888888'
    else:
        vcolor = l_alpha.get_color()
        
    for line in lines:
        if line['f_ik'] is not None and line['f_ik'] >= f_min:
            lamb_nm, gs, es, iso = line['lambda'] * 1e9, line['gs'], line['es'], line['isotope']
            level_text = iso + gs.replace('o', '/') + '->' + es.replace('o', '/')
            ax.axvline(x = lamb_nm, color = vcolor, linestyle = ':', alpha = 0.7)
            ax.text(lamb_nm, y_min * (0.9 - 0.1 * (text_ypos % 8)), level_text, color = vcolor, fontsize = 12)
            text_ypos += 1
    ax.text(0.98, 0.02, 'Transitions with ' + r'$f_{ki}>$' + '{} are labeled'.format(f_min), transform=ax.transAxes,
            horizontalalignment = 'right', verticalalignment = 'bottom')

#%% Calculate polarizibilities (This step is fast, plotting is the slow part)
alpha_s_K_4S1o2 = alpha_pol(0, wavelengths, K_4S1o2_LOI, '4S1o2')
alpha_s_Rb_5S1o2 = alpha_pol(0, wavelengths, Rb_5S1o2_LOI, '5S1o2')
alpha_s_K_4P1o2 = alpha_pol(0, wavelengths, K_4P1o2_LOI, '4P1o2')
alpha_s_K_4P3o2 = alpha_pol(0, wavelengths, K_4P3o2_LOI, '4P3o2')

# alpha_v_K_4S1o2 = alpha_pol(1, wavelengths, K_4S1o2_lines, '4S1o2')
alpha_v_K_4S1o2_F9o2 = alpha_pol(1, wavelengths, K_4S1o2_LOI, '4S1o2', F = 9/2, I = 4)
alpha_v_K_4S1o2_F7o2 = alpha_pol(1, wavelengths, K_4S1o2_LOI, '4S1o2', F = 7/2, I = 4)
alpha_v_Rb_5S1o2_F2 = alpha_pol(1, wavelengths, Rb_5S1o2_LOI, '5S1o2', F = 2, I = 3/2)
alpha_v_Rb_5S1o2_F1 = alpha_pol(1, wavelengths, Rb_5S1o2_LOI, '5S1o2', F = 1, I = 3/2)

avoas_K_4S1o2_F9o2 = alpha_v_K_4S1o2_F9o2/alpha_s_K_4S1o2
avoas_Rb_5S1o2_F2 = alpha_v_Rb_5S1o2_F2/alpha_s_Rb_5S1o2
avoas_Rb_5S1o2_F1 = alpha_v_Rb_5S1o2_F1/alpha_s_Rb_5S1o2

#%% Plot (bare polarizabilities)
### Scalar polarizability
ylim_s = 2e3
notable_f = 0.001
mark_line_bool = True

fig_alpha = plt.figure(0, figsize = (20,14))
fig_alpha.clf()
ax_s = fig_alpha.add_subplot(211)
ax_s.set_ylim(-ylim_s, ylim_s)
l_K_4S1o2 = ax_s.plot(wavelengths * 1e9, alpha_s_K_4S1o2 * pol_SI2au, label = r'$\alpha_s(K-4S_{1/2})$')
l_Rb_5S1o2 = ax_s.plot(wavelengths * 1e9, alpha_s_Rb_5S1o2 * pol_SI2au, label = r'$\alpha_s(Rb-5S_{1/2})$')
l_4P1o2 = ax_s.plot(wavelengths * 1e9, alpha_s_K_4P1o2 * pol_SI2au, label = r'$\alpha_s(K-4P_{1/2})$')
l_4P3o2 = ax_s.plot(wavelengths * 1e9, alpha_s_K_4P3o2 * pol_SI2au, label = r'$\alpha_s(K-4P_{3/2})$')
if mark_line_bool:
    mark_important_lines(ax_s, K_4S1o2_lines, notable_f, l_alpha = l_K_4S1o2[0])
    mark_important_lines(ax_s, Rb_5S1o2_lines, notable_f, l_alpha = l_Rb_5S1o2[0], init_text_height = 1)
    # mark_important_lines(ax_s, K_4P1o2_lines, notable_f, l_alpha = l_4P1o2[0], init_text_height = 4)
    # mark_important_lines(ax_s, K_4P3o2_lines, notable_f, l_alpha = l_4P3o2[0], init_text_height = 6)
ax_s.plot(wavelengths * 1e9, np.zeros_like(wavelengths), 'k--')
ax_s.axvline(x = lambda_sw * 1e9, color = 'green', alpha = 0.7, linestyle = '-')
ax_s.axvline(x = lambda_lw * 1e9, color = 'red', alpha = 0.7, linestyle = '-')
ax_s.set_title('Scalar polarizabilities (trap depth assumes a 1W Gaussian beam with beam waist = {:.1f} um)'.format(beam_waist * 1e6))
ax_s.set_ylabel(r'$\alpha$' + ' [a.u.]')
ax_s.legend()

# Add a secondary axis to calculate effective trap depth (at Gaussian beam center)
ryax_s = ax_s.secondary_yaxis('right', functions = (lambda x: x * alpha2uK, lambda x: x / alpha2uK))
ryax_s.set_ylabel('Trap depth [uK]')

### Vector polarizability
ylim_v = 20
mark_line_bool = False

ax_v = fig_alpha.add_subplot(212)
ax_v.set_ylim(-ylim_v, ylim_v)
# ax_v.plot(wavelengths * 1e9, alpha_v_K_4S1o2 * pol_SI2au, label = r'$\alpha_v(K-4S_{1/2})$')
ax_v.plot(wavelengths * 1e9, alpha_v_K_4S1o2_F9o2 * pol_SI2au, label = r'$\alpha_v(K-4S_{1/2}; F = 9/2)$')
ax_v.plot(wavelengths * 1e9, alpha_v_K_4S1o2_F7o2 * pol_SI2au, label = r'$\alpha_v(K-4S_{1/2}; F = 7/2)$')
ax_v.plot(wavelengths * 1e9, alpha_v_Rb_5S1o2_F2 * pol_SI2au, label = r'$\alpha_v(Rb-5S_{1/2}; F = 2)$')
ax_v.plot(wavelengths * 1e9, alpha_v_Rb_5S1o2_F1 * pol_SI2au, label = r'$\alpha_v(Rb-5S_{1/2}; F = 1)$')
if mark_line_bool:
    mark_important_lines(ax_v, K_4S1o2_lines, notable_f)
    mark_important_lines(ax_v, Rb_5S1o2_lines, notable_f, init_text_height = 2)
ax_v.plot(wavelengths * 1e9, np.zeros_like(wavelengths), 'k--')
ax_v.axvline(x = lambda_sw * 1e9, color = 'green', alpha = 0.7, linestyle = '-')
ax_v.axvline(x = lambda_lw * 1e9, color = 'red', alpha = 0.7, linestyle = '-')
ax_v.set_title('Vector polarizabilities')
ax_v.set_xlabel(r'$\lambda$' + ' [nm]')
ax_v.set_ylabel(r'$\alpha$' + ' [a.u.]')
ax_v.legend(loc = 'upper right')

# The factor of 1/2 is cancelled out by the difference in polarization
ryax_v = ax_v.secondary_yaxis('right', functions = (lambda x: x * alpha2kHz, lambda x: x / alpha2kHz))
ryax_v.set_ylabel(r'$\Delta_{\sigma^+ - \sigma^-}/m_F$' + ' [kHz]')

#%% Plot (things derived from alphas)
fig_r = plt.figure(1, figsize = (20,7))
fig_r.clf()
ax_r = fig_r.add_subplot(111)
ax_r.set_ylim(-0.2, 0.2)
ax_r.plot(wavelengths * 1e9, avoas_K_4S1o2_F9o2, label = r'$\alpha_v/\alpha_s(K-4S_{1/2}); F = 9/2$')
ax_r.plot(wavelengths * 1e9, avoas_Rb_5S1o2_F2, label = r'$\alpha_v/\alpha_s(Rb-5S_{1/2}); F = 2$')
ax_r.plot(wavelengths * 1e9, avoas_Rb_5S1o2_F1, label = r'$\alpha_v/\alpha_s(Rb-5S_{1/2}); F = 1$')
ax_r.plot(wavelengths * 1e9, np.zeros_like(wavelengths), 'k--')
ax_r.axvline(x = lambda_sw * 1e9, color = 'green', alpha = 0.7, linestyle = '-')
ax_r.axvline(x = lambda_lw * 1e9, color = 'red', alpha = 0.7, linestyle = '-')
ax_r.set_xlabel(r'$\lambda$' + ' [nm]')
ax_r.set_ylabel(r'$\alpha_v/\alpha_s$')
ax_r.legend()