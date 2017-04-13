# Example use script for csumuriaopt.py

"""
Examples for various types of analyses using csumuriaopt.py Classes.
"""

import csumuriaopt as ada
import numpy as np

# ------ Setup ------
np.set_printoptions(precision=4, suppress=True)         # printed output limited to 4 decimal places

# ------ New optical and aerosol analysis ------
# 'spam' is an object that allows for a specific analysis
spam = ada.OptAnalysis()            # will have 1 data point
# Generally speaking, each OptAnalysis() instance should be for analyses with
# the same number of data points. So create a second analysis here as well
eggs = ada.OptAnalysis()            # will have 3 data points


# ------ Creation of parameterized aerosol populations ------

# --- Aerosol population example 1: single mode ---
# - A new lognormally distributed aerosol population will be created based on parameterized values -
modes = 1                       # Number of modes in distribution
mu = 100.                       # Median diameter (nm)
gsd = 1.5                       # Geometric standard deviation
mf = 1.0                        # Modal number fraction
cn_conc = 500.                  # Total number concentration (#/cm^3)
kappa = 0.61                    # Modal kappa hygroscopicity parameter
dry_m = np.complex(1.54,0.002)  # Modal complex index of refraction
# - Setup variables -
# parameters for each mode are sent as arrays in the form:
parms = np.array([mu,gsd,mf])
# kappa and refractive index variables to be used by hygroscopicity and mie classes
var = {'kappa':kappa, 'dry_m':dry_m}
# - Crete a new aerosol population called 'pop1' using the cn_dist() method -
spam.cn_dist(name_ovr='pop1',
             num_parm=parms, CNconc=cn_conc, modes=modes,
             auto_bins=False, nbins=100,
             var=var)

# - Data and information about the new aerosol population can be accessed -
# The bin midpoint diameters
print spam.pop1.data.BinMid.d
# The bin dN/dlogDp values
print spam.pop1.data.dNdlogDp.d


# --- Aerosol population example 1: single mode ---
# - A new lognormally distributed aerosol population will be created based on parameterized values -
modes = 1                       # Number of modes in distribution
mu = 2769.6                       # Median diameter (nm)
gsd = 1.00000                       # Geometric standard deviation
mf = 1.0                        # Modal number fraction
vol_conc = 1.                  # Total number concentration (#/cm^3)
kappa = 0.61                    # Modal kappa hygroscopicity parameter
dry_m = np.complex(1.54,0.002)  # Modal complex index of refraction
# - Setup variables -
# parameters for each mode are sent as arrays in the form:
parms = np.array([mu,gsd,mf])
# kappa and refractive index variables to be used by hygroscopicity and mie classes
var = {'kappa':kappa, 'dry_m':dry_m}
# First create a custom set of bin midpoint diameters to use
Dp_low = 2769.6            # Smallest size bin midpoint
Dp_up = 2770.          # Largest size bin midpoint
nbins = 1              # number of bins to create
BinMid = np.logspace(np.log10(Dp_low),np.log10(Dp_up),nbins)    # a lognormally spaced set of bins
gvar = {'BinMid':BinMid}
# - Crete a new aerosol population called 'pop1' using the cn_dist() method -
spam.cn_dist(name_ovr='pop1',
             num_parm=parms, Volconc=vol_conc, modes=modes,
             auto_bins=False, nbins=1,
             gvar=gvar, var=var)

# --- Aerosol population example 2: multi-modal with specified bin midpoint diameters and multiple data points ---
# - A new lognormally distributed aerosol population will be created based on parameterized values -
modes = 2                       # Number of modes in distribution
mu = [60., 200.]                # Median diameter for each mode (nm)
gsd = [1.5, 1.65]               # Geometric standard deviation for each mode
mf = [0.25,0.75]                # Modal number fraction for each mode (should sum to 1)
ndp = 3                         # Number of data points in analysis
cn_conc = [150.,500.,5000.]     # Total number concentation (#/cm^3) for each data point
kappa = [[0.6, 0.2], [0.6, 0.2], [0.1,0.1]]                     # Modal kappa hygroscopicity parameter for each mode and each data point
dry_m = [[np.complex(1.54,0.002),np.complex(1.54,0.2)]]*ndp     # Modal complex index of refraction for each mode and each data point
# - Setup variables -
# parameters for each mode are sent as arrays in the form:
parms = np.array([mu[0],gsd[0],mf[0],mu[1],gsd[1],mf[1]])
# kappa and refractive index variables to be used by hygroscopicity and mie classes
var = {'kappa':kappa, 'dry_m':dry_m}
# - Crete a new aerosol population using the cn_dist() method -
# First create a custom set of bin midpoint diameters to use
Dp_low = 10.            # Smallest size bin midpoint
Dp_up = 1000.           # Largest size bin midpoint
nbins = 30              # number of bins to create
BinMid = np.logspace(np.log10(Dp_low),np.log10(Dp_up),nbins)    # a lognormally spaced set of bins
gvar = {'BinMid':BinMid}
# Instantiate the new CNdist instance as before, but in the 'eggs' analysis this time
eggs.cn_dist(name_ovr='pop2',
             num_parm=parms, CNconc=cn_conc, modes=modes,
             gen_dist=True, auto_bins=False,
             gvar=gvar, var=var)

# - Data and information about the new aerosol population can be accessed -
# The bin midpoint diameters
print eggs.pop2.data.BinMid.d
# The bin dN/dlogDp values for the first mode and first datapoint
print eggs.pop2.modes.m1.data.dNdlogDp.d[0]
# The bin dN/dlogDp values for the second mode and third datapoint
print eggs.pop2.modes.m2.data.dNdlogDp.d[2]


# ------ Creation of Optical reconstructions ------

# --- Optical reconstruction example 1: Pass aerosol population 1 for Mie optical reconstruction ---
# Include a relative humidity variable that gives the environmental RH
RH = 85.
var = {'RH':RH}
# - Create a new optical instance called 'opt1' -
spam.optical(cn_dist=spam.pop1, var=var, name_ovr='opt1')
# - Run an extinction reconstruction for the instance -
spam.opt1.ext_recon(varname='Ext_550_wet',              # name the variable to be reconstructed
                   wl=550.,                             # run at wavelength of 550 nm
                   ind_mix=True,                        # conduct volume averaging for refractive index
                   ret_ext=False,                       # calculate result based on scattering and absorption separately (True calculates extinction)
                   gf=True, save_gf=True,               # grow aerosol to equilibrium with RH and save resulting growth factor as a variable
                   set_dist=True)                       # create a new variable with the extinction distribution for each bin (db_ext/dlogDp)

# - Data and information about the new aerosol optical reconstruction can be accessed -
# print extinction coefficient at the data point
print spam.opt1.data.Ext_550_wet.d
# Growth factor for aerosol in each bin
print spam.opt1.data.gf.d


# --- Optical reconstruction example 2: Pass aerosol population 2 for Mie optical reconstruction ---
# Include a relative humidity variable that gives the environmental RH at each data point
RH = [85., 60., 96.]            # RH at each data point
var = {'RH':RH}
# - Create a new optical instance called 'opt2' -
eggs.optical(cn_dist=eggs.pop2, var=var, name_ovr='opt2')
# - Run an extinction reconstruction for the instance -
eggs.opt2.ext_recon(varname='Ext_550_wet',              # name the variable to be reconstructed
                    wl=550.,                            # run at wavelength of 550 nm
                    ind_mix=True,                       # conduct volume averaging for refractive index
                    ret_ext=True,                       # calculate result based on extinction
                    gf=True, save_gf=True,              # grow aerosol to equilibrium with RH and save resulting growth factor as a variable
                    set_dist=True)                      # create a new variable with the extinction distribution for each bin (db_ext/dlogDp)

# - Data and information about the new aerosol optical reconstruction can be accessed -
# print extinction coefficient at each data point
print eggs.opt2.data.Ext_550_wet.d
# print complex (scattering and absoprtion) coefficient distribution (at each bin midpoint diameter) at the first datapoint
print eggs.opt2.data.Ext_550_wet_dist.d[0]


# ------ AOD reconstruction using opt2 data points to form atmospheric column ------
# - Create a list of data points which form the path of interest -
# For a path (e.g. atmospheric column) consisting of each of the data points:
path_ind = [2,0,1]
# Grid edges for each data point in path_ind in meters
grid_low = [0.,1e3,5e3]
grid_up = [1e3,5e3,10e3]
# - Run the path AOD reconstruction -
eggs.opt2.path_AOD(AODname='AOD_1', ext_var='Ext_550_wet',
                   ind=path_ind, loc_center=False, loc_low=grid_low, loc_up=grid_up)

# - Data and information about the new aerosol optical reconstruction can be accessed -
# print AOD for the path
print eggs.opt2.data.AOD_1.AOD.d
# print AOD for each grid box
print eggs.opt2.data.AOD_1.AOD_grid.d
# print AOD fraction for each grid box
print eggs.opt2.data.AOD_1.AOD_frac.d


# ------ Run Mie reconstructions directly ------
# Single particle extinction method docstring for ext_calc()
"""Calculates extinction due to scattering and absorption using Mie theory.
Assumes spherical particles and cores/shells.
NOTE: the imaginary component of the index of refraction (absorption) should be
positive, e.g. opposite of actual value.
Arguments:
    N:      Number of particles of size Dp (#/cm^3)
    Dp:     Diameter of particles (nm) or shell diameter
    wl:     Wavelength of light for scattering calc (nm)
    m:      Complex refractive index or shell
    DpC:    Core Diameter of particles (nm) or None
    mC:     Core complex refractive index or None
    ret_ext:    If True, returns total extinction coefficient
                If False, returns tuple of (scattering, absorption) coefficients
Returns:
    Extinction Coefficient(s) in units of (Mm^-1)
Note: Explicit delegation of method from Optical() class.
"""
rslt = spam.ext_calc(N=100., Dp=200., wl=550., m=np.complex(1.54, 0.02))

# print extinction result
print rslt

# Single scatter albedo for the same case
rslt = spam.ssa_calc(Dp=200., wl=550., m=np.complex(1.54, 0.02))

print rslt

# Asymmetry parameter for the same case
rslt = spam.asy_calc(Dp=200., wl=550., m=np.complex(1.54, 0.02))

print rslt

# Aplitude scattering matrix components S1 and S2 for the same case
rslt = spam.S12_calc(Dp=200., wl=550., m=np.complex(1.54, 0.02))

print rslt
