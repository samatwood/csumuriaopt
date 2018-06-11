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
start = dt.datetime.now()

# - Directories -
base_dir = os.path.abspath(os.path.join(os.path.dirname(ada.__file__), os.pardir))
wrf_file_dir=os.path.join(base_dir,'WRF_model_files')
rams_file_dir=os.path.join(base_dir,'RAMS_model_files')
output_dir=os.path.join(base_dir,'AOD_output')
pop_opt_dir=os.path.join(base_dir,'model_pop')
dust_db_dir=os.path.join(base_dir,'dust_db')

# - Run parameters -
recompute_model_aerosol_types = False
run_RAMS = True
run_WRF = False
run_Both = False
plot_output = False
# For memory limited issues
separate_pop_types = True

# - Reconstruction parameters -
# Wavelength (nm)
wl = 550.

# - Aerosol Population Type usage -
# All RAMS aerosol types
RAMS_pop_types = ['RAMS_salt_film', 'RAMS_salt_jet', 'RAMS_salt_spume',
                  'RAMS_dust1', 'RAMS_dust2',
                  'RAMS_ccn', 'RAMS_regen_aero1', 'RAMS_regen_aero2']
RAMS_pop_type_to_model_var = dict(
    RAMS_salt_film='salt_film_mass',
    RAMS_salt_jet='salt_jet_mass',
    RAMS_salt_spume='salt_spume_mass',
    RAMS_dust1='dust1_mass',
    RAMS_dust2='dust2_mass',
    RAMS_ccn='ccn_mass',
    RAMS_regen_aero1='regen_aero1_mass',
    RAMS_regen_aero2='regen_aero2_mass'
)

# RAMS_pop_types = ['RAMS_salt_film_alt', 'RAMS_salt_jet_alt', 'RAMS_salt_spume_alt']
# RAMS_pop_type_to_model_var = dict(
#     RAMS_salt_film_alt='salt_film_mass',
#     RAMS_salt_jet_alt='salt_jet_mass',
#     RAMS_salt_spume_alt='salt_spume_mass'
# )

# All_salt_pop_types = ['RAMS_salt_film', 'RAMS_salt_jet', 'RAMS_salt_spume',
#                       'WRF_SEAS_1', 'WRF_SEAS_2', 'WRF_SEAS_3', 'WRF_SEAS_4']
# All_salt_pop_type_to_model_var = dict(
#     RAMS_salt_film='salt_film_mass',
#     RAMS_salt_jet='salt_jet_mass',
#     RAMS_salt_spume='salt_spume_mass',
#
#     WRF_SEAS_1='SEAS_1',
#     WRF_SEAS_2='SEAS_2',
#     WRF_SEAS_3='SEAS_3',
#     WRF_SEAS_4='SEAS_4',
# )


RAMS_model_var_conc_type = 'mass'


# ------ Model Aerosol Population Types ------
if recompute_model_aerosol_types:

    spam = model_pop(wl, pop_opt_dir, ret=True)

    # --- Example Usage ---
    if False:
        # Calculate extinction coefficient for the WRF_SEAS_1 population type in a grid box with
        #   1000 cm^-3 number concentration and 80% RH
        spam.WRF_SEAS_1_550.opt.ext_cn(80, 1000)
        spam.WRF_SEAS_2_550.opt.ext_cn(80, 50)
        spam.WRF_SEAS_1_550.opt.ext_vol(80, 10)
        spam.WRF_SEAS_2_550.opt.ext_vol(80, 10)
        spam.WRF_SEAS_1_550.opt.ext_mass(80, 15, 1.5)
        spam.WRF_SEAS_2_550.opt.ext_mass(80, 15, 1.5)

    end = dt.datetime.now()
    print(end-start)


# ------ New model optical and aerosol analysis ------

# RAMS model
if run_RAMS:
    spam = ada.RAMS(wl=550.,
                    model_file_dir=rams_file_dir,
                    output_dir=output_dir,
                    pop_opt_dir=pop_opt_dir,
                    dust_db_dir=dust_db_dir,
                    default_pop_types = RAMS_pop_types,
                    pop_type_to_model_var = RAMS_pop_type_to_model_var,
                    model_var_conc_type=RAMS_model_var_conc_type,
                    process_all_files=True,
                    plotting=plot_output,
                    separate_pop_types=separate_pop_types
                    )

# WRF model
if run_WRF:
    eggs = ada.WRF(model_file_dir=wrf_file_dir,
                   output_dir=output_dir,
                   pop_opt_dir=pop_opt_dir,
                   dust_db_dir=dust_db_dir
                   )

# Both - Testing
if run_Both:
    spam = ada.RAMS(wl=550.,
                    model_file_dir=rams_file_dir,
                    output_dir=output_dir,
                    pop_opt_dir=pop_opt_dir,
                    dust_db_dir=dust_db_dir,
                    default_pop_types = All_salt_pop_types,
                    pop_type_to_model_var = All_salt_pop_type_to_model_var,
                    model_var_conc_type=RAMS_model_var_conc_type,
                    process_all_files=True,
                    plotting=plot_output,
                    separate_pop_types=separate_pop_types
                    )


end = dt.datetime.now()
print(end-start)


# Plot methods
if plot_output:
    # spam._plot.pop_AOD_example()
    spam._plot.pop_fRH()
