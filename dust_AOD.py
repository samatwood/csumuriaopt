# author: Emily Bian, Sam Atwood
# email: bianqj@atmos.colostate.edu, satwood@atmos.colostate.edu

"""
As Yang's database does not cover all the size bins from WRF-CHEM, I have to re-calculate effective radius
and lognormal variance for all the dust particles throughout 5 size bins in each grid.
The code is designed to calculate the minimum and maximum AOD using the output from WRF-CHEM.
-Emily Bian, 01/03/2018

Modifications to allow inclusion in csumuiraopt: Sam Atwood, 25JAN2018
"""
import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

############ read Yang's database ############################
dust_db_dir = './dust_db/'
WRF_files_dir = './WRF_model_files/'
# - File Names -
# Aspect Ratio
aspect_ratio_fn = ['1.02', '1.05', '1.10', '1.13', '1.16', '1.20', '1.50', '1.80', '2.10', '2.40', '2.70', '3.30']
# Prolate Shape
prolate_fn = os.path.join(dust_db_dir, 'Spheroid_prolate', 'Bulk_prolate_AspR_')
# Oblate Shape
oblate_fn = os.path.join(dust_db_dir, 'Spheroid_oblate', 'Bulk_oblate_AspR_')
# Optical Properties
subfile = os.sep + 'isca.dat'  # optical properties

# wl = np.loadtxt('wl.txt', delimiter=' ', skiprows=0)  # um, load wavelength information
wl = np.array([0.45,0.55,0.70])
rff = np.array([1.2, 1.5, 2., 2.5, 3.])  # um, effective radius in Yang's database
variance = np.arange(1.2, 3.2, 0.2)
pi = 3.1415926

####################################################################
#               read WRF-CHEM output                               #
####################################################################

dust_density = 2.2  # g/cm3, assume dust density from WRF-CHEM output
wc_filename = 'd01_2016-08-04_12_00_00'  # ncf
wrf_chem_path = WRF_files_dir + 'wrfout_'

root_grp = Dataset(wrf_chem_path + wc_filename)
lat = root_grp.variables['XLAT'][:]
long = root_grp.variables['XLONG'][:]
dustload1 = root_grp.variables['DUSTLOAD_1'][:]
dustload2 = root_grp.variables['DUSTLOAD_2'][:]
dustload3 = root_grp.variables['DUSTLOAD_3'][:]
dustload4 = root_grp.variables['DUSTLOAD_4'][:]
dustload5 = root_grp.variables['DUSTLOAD_5'][:]
root_grp.close()
dustvol1 = dustload1 / dust_density * 1e-6
dustvol2 = dustload2 / dust_density * 1e-6
dustvol3 = dustload3 / dust_density * 1e-6
dustvol4 = dustload4 / dust_density * 1e-6
dustvol5 = dustload5 / dust_density * 1e-6

dustvol = dustvol1 + dustvol2 + dustvol3 + dustvol4 + dustvol5

dustarea1 = 3. / 4. * dustvol1 / rff[0]
dustarea2 = 3. / 4. * dustvol2 / rff[1]
dustarea3 = 3. / 4. * dustvol3 / rff[2]
dustarea4 = 3. / 4. * dustvol4 / rff[3]
dustarea5 = 3. / 4. * dustvol5 / rff[4]

dustarea = dustarea1 + dustarea2 + dustarea3 + dustarea4 + dustarea5
dustarea = np.squeeze(dustarea)
# effective radius for the whole particles
Reff = 3. / 4. * dustvol / dustarea

vff = 0.2  # effective variance for each size bin
m4_1 = (vff + 1) * (3 * dustvol1) ** 2 / (4 * pi) ** 2 / (dustarea1 / pi)
m4_2 = (vff + 1) * (3 * dustvol2) ** 2 / (4 * pi) ** 2 / (dustarea2 / pi)
m4_3 = (vff + 1) * (3 * dustvol3) ** 2 / (4 * pi) ** 2 / (dustarea3 / pi)
m4_4 = (vff + 1) * (3 * dustvol4) ** 2 / (4 * pi) ** 2 / (dustarea4 / pi)
m4_5 = (vff + 1) * (3 * dustvol5) ** 2 / (4 * pi) ** 2 / (dustarea5 / pi)

m4_t = m4_1 + m4_2 + m4_3 + m4_4 + m4_5

vff_t = (dustarea / pi) * m4_t / (3. * dustvol / 4 / pi) ** 2 - 1
# convert to the variation of lognormal distribution
variance_t = np.exp((np.log(1 + vff_t)) ** 0.5)

vff_t = vff_t.squeeze()
variance_t = variance_t.squeeze()

#############################################################################
#                       load light scattering data                          #
#############################################################################

prolate1 = np.loadtxt(prolate_fn + aspect_ratio_fn[0] + subfile, delimiter=' ', skiprows=0)
prolate2 = np.loadtxt(prolate_fn + aspect_ratio_fn[1] + subfile, delimiter=' ', skiprows=0)
prolate3 = np.loadtxt(prolate_fn + aspect_ratio_fn[2] + subfile, delimiter=' ', skiprows=0)
prolate4 = np.loadtxt(prolate_fn + aspect_ratio_fn[3] + subfile, delimiter=' ', skiprows=0)
prolate5 = np.loadtxt(prolate_fn + aspect_ratio_fn[4] + subfile, delimiter=' ', skiprows=0)
prolate6 = np.loadtxt(prolate_fn + aspect_ratio_fn[5] + subfile, delimiter=' ', skiprows=0)
prolate7 = np.loadtxt(prolate_fn + aspect_ratio_fn[6] + subfile, delimiter=' ', skiprows=0)
prolate8 = np.loadtxt(prolate_fn + aspect_ratio_fn[7] + subfile, delimiter=' ', skiprows=0)
prolate9 = np.loadtxt(prolate_fn + aspect_ratio_fn[8] + subfile, delimiter=' ', skiprows=0)
prolate10 = np.loadtxt(prolate_fn + aspect_ratio_fn[9] + subfile, delimiter=' ', skiprows=0)
prolate11 = np.loadtxt(prolate_fn + aspect_ratio_fn[10] + subfile, delimiter=' ', skiprows=0)
prolate12 = np.loadtxt(prolate_fn + aspect_ratio_fn[11] + subfile, delimiter=' ', skiprows=0)
prolate = np.concatenate((prolate1, prolate2, prolate3, prolate4, prolate5, prolate6,
                          prolate7, prolate8, prolate9, prolate10, prolate11, prolate12), axis=0)
oblate1 = np.loadtxt(oblate_fn + aspect_ratio_fn[0] + subfile, delimiter=' ', skiprows=0)
oblate2 = np.loadtxt(oblate_fn + aspect_ratio_fn[1] + subfile, delimiter=' ', skiprows=0)
oblate3 = np.loadtxt(oblate_fn + aspect_ratio_fn[2] + subfile, delimiter=' ', skiprows=0)
oblate4 = np.loadtxt(oblate_fn + aspect_ratio_fn[3] + subfile, delimiter=' ', skiprows=0)
oblate5 = np.loadtxt(oblate_fn + aspect_ratio_fn[4] + subfile, delimiter=' ', skiprows=0)
oblate6 = np.loadtxt(oblate_fn + aspect_ratio_fn[5] + subfile, delimiter=' ', skiprows=0)
oblate7 = np.loadtxt(oblate_fn + aspect_ratio_fn[6] + subfile, delimiter=' ', skiprows=0)
oblate8 = np.loadtxt(oblate_fn + aspect_ratio_fn[7] + subfile, delimiter=' ', skiprows=0)
oblate9 = np.loadtxt(oblate_fn + aspect_ratio_fn[8] + subfile, delimiter=' ', skiprows=0)
oblate10 = np.loadtxt(oblate_fn + aspect_ratio_fn[9] + subfile, delimiter=' ', skiprows=0)
oblate11 = np.loadtxt(oblate_fn + aspect_ratio_fn[10] + subfile, delimiter=' ', skiprows=0)
oblate12 = np.loadtxt(oblate_fn + aspect_ratio_fn[11] + subfile, delimiter=' ', skiprows=0)
oblate = np.concatenate((oblate1, oblate2, oblate3, oblate4, oblate5, oblate6,
                         oblate7, oblate8, oblate9, oblate10, oblate11, oblate12), axis=0)
combine_po = np.concatenate((prolate, oblate), axis=0)
# AERONET wavelength: 440, 673, 871, 1020 nm
wl_t = 10.35  # um
wl_min = abs(wl - wl_t)
[wlr, ] = np.where(wl_min == min(wl_min))
print wl[wlr[0]]

Reff = np.squeeze(Reff)
a, b = Reff.shape
qext_min = np.zeros([a, b])
qext_max = np.zeros([a, b])
tau_min = np.zeros([a, b])
tau_max = np.zeros([a, b])

for k in range(0, a):
    print k
    for j in range(0, b):
        Reff_min = abs(rff - Reff[k, j])
        [Rfr, ] = np.where(Reff_min == min(Reff_min))  # find the corresponding effective radius
        variance_min = abs(variance - variance_t[k, j])
        [Var, ] = np.where(variance_min == min(variance_min))  # find the corresponding effective radius
        temp1 = combine_po[combine_po[:, 0] == wl[wlr[0]]]
        temp2 = temp1[temp1[:, 1] == rff[Rfr[0]]]
        temp3 = temp2[temp2[:, 2] == round(variance[Var[0]], 1)]
        qext = temp3[:, 5]  # Qext for the corresponding size of dust particle
        qext_min[k, j] = np.min(qext)  # minimum qext
        qext_max[k, j] = np.max(qext)  # maxmium qext
        tau_min[k, j] = qext_min[k, j] * dustarea[k, j]  # minimum AOD
        tau_max[k, j] = qext_max[k, j] * dustarea[k, j]  # maximum AOD
np.savetxt(wc_filename + '_taumin.txt', tau_min)  # output minimum AOD
np.savetxt(wc_filename + '_taumax.txt', tau_max)  # output max AOD
# np.savetxt(wc_filename+'_qext.txt',qext)

if tau_min.max() == 0.0:
    tau_min = np.loadtxt(wc_filename + '_taumin.txt')
    tau_max = np.loadtxt(wc_filename + '_taumax.txt')

# plot min AOD
plt.figure(1)
lat = np.squeeze(lat)
long = np.squeeze(long)
m = Basemap(llcrnrlon=32, llcrnrlat=12, urcrnrlon=65, urcrnrlat=40, projection='mill', resolution='l')
# draw coastlines.
m.drawcoastlines(linewidth=1.25)
# draw a boundary around the map, fill the background.
m.drawmapboundary()
m.drawcountries()
# clevs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
cs = m.pcolor(long, lat, tau_min, cmap=plt.cm.rainbow, latlon=True)
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('AOD')
plt.show()

# plot max AOD
plt.figure(2)
m = Basemap(llcrnrlon=32, llcrnrlat=12, urcrnrlon=65, urcrnrlat=40, projection='mill', resolution='l')
# draw coastlines.
m.drawcoastlines(linewidth=1.25)
# draw a boundary around the map, fill the background.
m.drawmapboundary()
m.drawcountries()
# clevs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
cs = m.pcolor(long, lat, tau_max, cmap=plt.cm.rainbow, latlon=True)
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('AOD')
plt.show()
