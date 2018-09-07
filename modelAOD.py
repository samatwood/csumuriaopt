# Example use script for csumuriaopt.py for a rough reconstruction of model AOD for WRF-CHEM and RAMS

import os
import numpy as np
import scipy as sp
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import datetime as dt
from mpl_toolkits.basemap import Basemap
import csumuriaopt as ada
from csumuriaopt.model_pop import model_pop


# ------ Setup ------
np.set_printoptions(precision=4, suppress=True)         # printed output limited to 4 decimal places
# start = dt.datetime.now()

# - Directories -
base_dir = os.path.abspath(os.path.join(os.path.dirname(ada.__file__), os.pardir))
wrf_file_dir=os.path.join(base_dir,'WRF_model_files')
rams_file_dir=os.path.join(base_dir,'RAMS_model_files')
output_dir=os.path.join(base_dir,'AOD_output')
pop_opt_dir=os.path.join(base_dir,'model_pop')
dust_db_dir=os.path.join(base_dir,'dust_db')

# - Pre-processing options -
recompute_model_aerosol_types = False   # Computes stored aerosol lookup tables (only need to do once)
                                        # All aerosol population types listed in model_pop.py will be computed
                                        # NOTE: Population parameters may need to be changed for different wavelengths

# - Run options -
run_RAMS = True                         # Run RAMS aerosol AOD computation
run_WRF = False                         # Run WRF aerosol AOD computation (Not yet setup)

# - File output options -
save_hdf5 = True                        # Save hdf5 output files
save_netcdf4 = False                    # Save netCDF4 output files
compress_output = False                 # Compress output files (only applies to hdf5 files currently)
separate_pop_types = True               # Run and save AOD for each aerosol type separately in memory limited conditions

# - Plotting options -
plot_output = True                      # Plot sample dry and wet AOD and size distributions for each population type
plot_existing = False                   # Plot existing hdf5 output files as separate population types (no model run)
plot_fRH = False                        # Plot fRH curves for population types (no model run)
                                        # NOTE: This doesn't yet work for pop types with variable median diameters

# - Reconstruction model parameters -
cap_RH = True                           # Cap RH values to < 100% to prevent cloud masking
wl = 550.                               # Wavelength for AOD/extinction computation (nm)

# - Grid parameters -
# RAMS model top (meters) for computing sigma-z levels over topography (found in header files)
rams_ztop = 21996.
# RAMS model level grid centers on ZT levels (include ALL levels here and in REVU output data)
rams_levs = np.array([
    -36, 36, 114, 198, 289, 387, 494, 608, 732, 865,
    1010, 1165, 1334, 1515, 1712, 1924, 2153, 2400, 2667, 2955,
    3267, 3603, 3966, 4359, 4782, 5240, 5734, 6268, 6845, 7467,
    8140, 8867, 9621, 10371, 11121, 11871, 12621, 13371, 14121, 14871,
    15621, 16371, 17121, 17871, 18621, 19371, 20121, 20871, 21621, 22371,
    23121])

# - Aerosol Population Types -
# Available RAMS aerosol types
# 'RAMS_salt_film', 'RAMS_salt_jet','RAMS_salt_spume','RAMS_dust1',
# 'RAMS_dust2','RAMS_ccn','RAMS_regen_aero1','RAMS_regen_aero2'

RAMS_pop_types = ['RAMS_salt_film', 'RAMS_salt_jet', 'RAMS_salt_spume',
                  'RAMS_dust1', 'RAMS_dust2',
                  'RAMS_ccn',
                  'RAMS_regen_aero1', 'RAMS_regen_aero2']

# Available WRF-CHEM aerosol types
# 'WRF_SEAS_1', 'WRF_SEAS_2', 'WRF_SEAS_3', 'WRF_SEAS_4'

WRF_pop_types = ['WRF_SEAS_1', 'WRF_SEAS_2', 'WRF_SEAS_3', 'WRF_SEAS_4']


# ------ Model Aerosol Population Types ------
if recompute_model_aerosol_types:

    spam = model_pop(wl, pop_opt_dir, ret=True)

    # --- Example Usage ---
    # # Calculate extinction coefficient for the WRF_SEAS_1 population type in a grid box with
    # #   1000 cm^-3 number concentration and 80% RH
    # spam.WRF_SEAS_1_550.opt.ext_cn(80, 1000)
    # spam.WRF_SEAS_2_550.opt.ext_cn(80, 50)
    # spam.WRF_SEAS_1_550.opt.ext_vol(80, 10)
    # spam.WRF_SEAS_2_550.opt.ext_vol(80, 10)
    # spam.WRF_SEAS_1_550.opt.ext_mass(80, 15, 1.5)
    # spam.WRF_SEAS_2_550.opt.ext_mass(80, 15, 1.5)

    # end = dt.datetime.now()
    # print(end-start)


# ------ New model optical and aerosol analysis ------

# RAMS model
if run_RAMS:
    spam = ada.RAMS(wl=wl,
                    model_file_dir=rams_file_dir,
                    output_dir=output_dir,
                    pop_opt_dir=pop_opt_dir,
                    dust_db_dir=dust_db_dir,
                    grid_nominal_centers=rams_levs,
                    grid_ztop=rams_ztop,
                    pop_types = RAMS_pop_types,
                    separate_pop_types=separate_pop_types,
                    cap_RH = cap_RH,
                    plot_output=plot_output,
                    plot_existing = plot_existing,
                    plot_fRH = plot_fRH,
                    save_hdf5 = save_hdf5,
                    save_netcdf4 = save_netcdf4,
                    compression = compress_output
                    )

# WRF model
if run_WRF:
    eggs = ada.WRF(model_file_dir=wrf_file_dir,
                   output_dir=output_dir,
                   pop_opt_dir=pop_opt_dir,
                   dust_db_dir=dust_db_dir
                   )


# end = dt.datetime.now()
# print(end-start)

# - Additional Plot methods -
# spam._plot.pop_AOD_example()
# spam._plot.pop_fRH()
