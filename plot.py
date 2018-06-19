# plot.py

import os
import numpy as np
import matplotlib.colors as mplc
import matplotlib.pylab as pl
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
# Basemap not always installed
try:
    from mpl_toolkits.basemap import Basemap
    bmf = True
except ImportError:
    bmf = False


class Plotting(object):
    def __init__(self, parent):
        self.parent = parent

    def _plot_list(self, ind, x, y=None, z=None):
        '''Sets up a list of variables for each set of data to be plotted.
        ind determines the number of plots and therefore the length of the x,y,z lists.
        If ind is None, then x,y,z are either single sets of data to be plotted or
        are lists which should be the same size.
        Returns (n,ind,x(,y,z)) as a tuple where n is the number of plot sets,
        and ind,x,y,z are lists of length n.
        '''
        if ind is not None:
            # ind determines the number of plot sets
            if not isinstance(ind, list):
                n = 1
            else:
                n = len(ind)
        else:
            # ind does not determine the number of plot sets
            if not isinstance(x, list):
                n = 1
            else:
                n = len(x)
        # set lists of proper length
        if not isinstance(ind, list):
            ind = [ind]*n
        if not isinstance(x, list):
            x = [x]*n
        if len(x) != n:
            raise Exception('x var of length %i, is not equal to plot set size of %i'%(len(x),n))
        if y is not None:
            if not isinstance(y, list):
                y = [y]*n
            if len(y) != n:
                raise Exception('y var of length %i, is not equal to plot set size of %i'%(len(y),n))
        if z is not None:
            if not isinstance(z, list):
                z = [z]*n
            if len(y) != n:
                raise Exception('z var of length %i, is not equal to plot set size of %i'%(len(z),n))
        # return all variables
        return (n,ind,x,y,z)

    def _plot_vars(self, x, y=None, z=None, xind=None, yind=None, zind=None, ztrans=False,
                   use_only_valid=True, cax=None):
        '''Sets up and preps variables for a new plot on top of a figure and axis.
        Arguments:
            x:              The first variable as numpy array or VarObject
            y:              Optional second variable as numpy array or VarObject
            z:              Optional third plot variable as numpy array or VarObject
            x|y|zind:       The indicies to plot for each plot variable
            cax:            The axis to plot on
            ztrans:         If True, will transpose the value of z right before returning it
            use_only_valid: If True, will replace any datapoints with a False .v value with
                            NaN, 0, or False, depending on array datatype
                            Note: This is only conducted on variables that are VarObjects
            cax:            The current axis, which is set to the current axis
        Returns x,y,z values as numpy arrays and x,y,z booleans that are True if each are VarObjects.
        All int arrays will be converted to float.
        '''
        pl.sca(cax)
        ret_list = []
        # Determine each variable type
        if isinstance(x, da.DataClasses.VarObject):
            xvar=True
            if use_only_valid:
                val = np.ma.masked_array(x.d, mask=~x.v, fill_value=self._get_array_null_value(x.d.dtype)).filled()
            else:
                val = x.d
            if xind is not None:
                x_ = val[xind]
            else:
                x_ = val
        else:
            xvar=False
            if xind is not None:
                x_ = x[xind]
            else:
                x_ = x

        if y is not None:
            if isinstance(y, da.DataClasses.VarObject):
                yvar=True
                if use_only_valid:
                    val = np.ma.masked_array(y.d, mask=~y.v, fill_value=self._get_array_null_value(y.d.dtype)).filled()
                else:
                    val = y.d
                if yind is not None:
                    y_ = val[yind]
                else:
                    y_ = val
            else:
                yvar=False
                if yind is not None:
                    y_ = y[yind]
                else:
                    y_ = y
        else:
            yvar = None
            y_ = None

        if z is not None:
            if isinstance(z, da.DataClasses.VarObject):
                zvar=True
                if use_only_valid:
                    val = np.ma.masked_array(z.d, mask=~z.v, fill_value=self._get_array_null_value(z.d.dtype)).filled()
                else:
                    val = z.d
                if zind is not None:
                    z_ = val[zind]
                else:
                    z_ = val
            else:
                zvar=False
                if zind is not None:
                    z_ = z[zind]
                else:
                    z_ = z
        else:
            zvar = None
            z_ = None
        if ztrans:
            z_ = np.transpose(z_)

        # Convert to numpy arrays and int arrays to float arrays
        x_ = np.array(x_)
        y_ = np.array(y_)
        z_ = np.array(z_)
        if x_.dtype is np.dtype('int'):
            x_ = np.array(x_, dtype=float)
        if y_.dtype is np.dtype('int'):
            y_ = np.array(y_, dtype=float)
        if z_.dtype is np.dtype('int'):
            z_ = np.array(z_, dtype=float)

        # Create return list
        ret_list.append(x_)
        if y is not None:
            ret_list.append(y_)
        if z is not None:
            ret_list.append(z_)
        ret_list.append(xvar)
        if y is not None:
            ret_list.append(yvar)
        if z is not None:
            ret_list.append(zvar)
        # Return
        return ret_list

    def _plot_format(self, ax, x=None, y=None, z=None, xvar=None, yvar=None, zvar=None,
                     xlim=None, ylim=None, xlog=None, ylog=None, xmin=None, ymin=None,
                     xtimeline=False, majl=None, minl=None, ymajl=None, yminl=None,
                     xformat=None, yformat=None, yminorformat=None,
                     xticks=None, xtickmajorlength=5, xtickmajorwidth=1, xtickminorlength=3, xtickminorwidth=1,
                     yticks=None, ytickmajorlength=5, ytickmajorwidth=1, ytickminorlength=3, ytickminorwidth=1,
                     xgridmaj='--', xgridmin=':', ygridmaj=True, ygridmin=None,
                     xvis=None, yvis=None,
                     xlabel=True, ylabel=True, zlabel=True, tsformat='%d %b %Y\n%H:%M UTC',
                     plot_letter=None, pl_loc=[0.01, 0.96],
                     legend=None, ncol=1, lsize=10, borderpad=0.5,
                     cb=False, cb_ticks=None, cb_format=None, cb_title=None, cb_map=None, extend='neither',
                     orientation='vertical',
                     title=None, **kwargs):
        '''

        Notes for later
        extend options:     'max', 'min', 'both', 'neither'
        cb options:     True: new colorbar, False: no colobar, axes: location of new cb axis
        x|ylabel:       If True, will grab from VarObject if available, otherwise will set to value
        xtimeline:      If True, will format x variable as a timeline
        plot_letter:    If not None, a letter that is overlaid on the plot
        pl_loc:         The location of the plot letter
        '''
        # Make ax the current axis
        # pl.sca(ax)
        # Set axes limits and scale
        if xlim is not None:
            pl.xlim(xlim)
        if ylim is not None:
            pl.ylim(ylim)
        if xlog:
            pl.xscale('log')
        if ylog:
            pl.yscale('log')
        if xmin is not None:
            pl.xlim(xmin=xmin)
        if ymin is not None:
            pl.ylim(ymin=ymin)
        # Set Ticks
        if xticks is not None:
            pl.xticks(xticks)
        ax.xaxis.set_tick_params(length=xtickmajorlength, width=xtickmajorwidth, which='major')
        ax.xaxis.set_tick_params(length=xtickminorlength, width=xtickminorwidth, which='minor')
        if yticks is not None:
            pl.yticks(yticks)
        ax.yaxis.set_tick_params(length=ytickmajorlength, width=ytickmajorwidth, which='major')
        ax.yaxis.set_tick_params(length=ytickminorlength, width=ytickminorwidth, which='minor')
        if majl is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(majl))
        if minl is not None:
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(minl))
        if ymajl is not None:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ymajl))
        if yminl is not None:
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(yminl))
        # If x axis is a timeline, format for date, otherwise set axis tick label formats
        if xtimeline:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(tsformat))
        else:
            if xformat is not None:
                ax.xaxis.set_major_formatter(xformat)
        if yformat is not None:
            ax.yaxis.set_major_formatter(yformat)
        if yminorformat is not None:
            ax.yaxis.set_minor_formatter(yminorformat)
        # Set axis tick label visibility and grid
        if xvis is not None:
            pl.setp(ax.get_xticklabels(), visible=xvis)
        if yvis is not None:
            pl.setp(ax.get_yticklabels(), visible=yvis)
        if xgridmaj:
            if xgridmaj is True:
                ax.xaxis.grid(True)
            else:
                ax.xaxis.grid(True, which='major', linestyle=xgridmaj)
        if xgridmin:
            if xgridmin is True:
                ax.xaxis.grid(True)
            else:
                ax.xaxis.grid(True, which='minor', linestyle=xgridmin)
        if ygridmaj:
            if ygridmaj is True:
                ax.yaxis.grid(True)
            else:
                ax.yaxis.grid(True, which='major', linestyle=ygridmaj)
        if ygridmin:
            if ygridmin is True:
                ax.yaxis.grid(True)
            else:
                ax.yaxis.grid(True, which='minor', linestyle=ygridmin)
        # Set axis labels
        if xlabel is True:
            if xvar is True:
                pl.xlabel(x.label_axis + ' ' + x.units)
        elif xlabel is not False:
            pl.xlabel(xlabel)
        if ylabel is True:
            if yvar is True:
                pl.ylabel(y.label_axis + ' ' + y.units)
        elif ylabel is not False:
            pl.ylabel(ylabel)
        if legend is not None:
            pl.legend(loc=legend, labelspacing=0, prop={'size': lsize}, ncol=ncol, numpoints=1, borderpad=borderpad)
        # Add plot letter
        if plot_letter is not None:
            pl.text(pl_loc[0], pl_loc[1], '%s' % plot_letter, ha='left', va='top',
                    fontdict={'size': 16, 'weight': 'bold'}, transform=ax.transAxes,
                    bbox={'alpha': 0.35, 'color': 'w'})
        # Colorbar
        if cb is True:
            cbobj = pl.colorbar(mappable=cb_map, ticks=cb_ticks, format=cb_format,
                                extend=extend, orientation=orientation)
            cbobj.set_label(cb_title)
        #            if zvar is True:
        #                cbobj.set_label(z.label_axis+' '+z.units)
        elif cb is False:
            pass
        else:
            cbax = pl.axes(cb)
            cbobj = pl.colorbar(mappable=cb_map, cax=cbax, ticks=cb_ticks, format=cb_format,
                                extend=extend, orientation=orientation)
            if cb_title is not None:
                cbobj.set_label(cb_title)
        #            if zlabel is True:
        #                if zvar is True:
        #                    cbobj.set_label(z.label_axis+' '+z.units)
        #            elif zlabel is not False:
        #                cbobj.set_label(zlabel)
        pl.sca(ax)
        # Set title
        if title is not None:
            pl.title(title)

    def simple_dist_plot(self, cnobj, ind, xlim=None, ls='-', c='r', label=None,
                         plot_voldist=True, dmy=False, ax=None, title=None):
        # Plot setup
        if ax is None:
            fig = pl.figure(figsize=(8 ,6))
            # pl.subplots_adjust(bottom=0.125, top=0.925, right=0.75)
            ax =pl.subplot(111)
        else:
            pl.sca(ax)
        # Set up full distribution data by adding the last BinUp size to BinLow array
        x_ = np.append(cnobj.data.BinLow.d ,cnobj.data.BinUp.d[-1])
        y_ = cnobj.data.Norm_dNdlogDp.d[ind]
        yp_ = np.append(y_,y_[-1])
        y2_ = cnobj.data.Norm_dVdlogDp.d[ind]
        yp2_ = np.append(y2_,y2_[-1])
        if plot_voldist:
            pl.step(x_, yp_, color='r', where='post', linestyle=ls, lw=1, label='Number')
            pl.step(x_, yp2_, color='b', where='post', linestyle=ls, lw=1, label='Volume')
            # pl.title(cnobj.__name__)
            pl.legend(loc=0)
            if xlim is not None:
                pl.xlim(xlim)
            # Format plot
            self._plot_format(ax, x=cnobj.data.BinMid, xvar=True, ylabel='Normalized ${dVar/dlogD_p}$',
                              xlog=True, xgridmin=None, title=title)
        else:
            pl.step(x_, yp_, color=c, where='post', linestyle=ls, lw=1, label=label)
            # Format plot
            if dmy == 0:
                self._plot_format(ax, x=cnobj.data.BinMid, xvar=True, ylabel='Normalized ${dVar/dlogD_p}$',
                              xlog=True, xgridmin=None, title=title)

        # pl.show()

    def _auto_extents(self):
        bottom = self.parent._data.lat.min()
        top = self.parent._data.lat.max()
        left = self.parent._data.lon.min()
        right = self.parent._data.lon.max()
        # if bottom < -90.:
        #     bottom = -90.
        # if top > 90.:
        #     top = 90.
        # if left < -180.:
        #     left += 360.
        # if right > 180:
        #     right -= 360.
        return [bottom, left, top, right]

    def _map_setup(self, ax, map_extents=None, zorder=0, **kwargs):
        """Sets up a BaseMap figure.
        Returns the figure or axes object, and map object as tuple, (fig, m)
        Can setup a map an existing figure with cax.
        Parameters:
            map_extents:    Map corners as [ll_lat,ll_lon, ur_lat,ur_lon]
                            Lat(-90:90), Lon(-180:180)
            figsize:        Figure size
            ax:             The axes to draw the basemap instance on
            zorder:         Will draw map components at this zorder
        """
        if map_extents is None:
            map_extents = self._auto_extents()
        m = Basemap(projection='cyl', resolution='l',
                    llcrnrlat=map_extents[0], urcrnrlat=map_extents[2],
                    llcrnrlon=map_extents[1], urcrnrlon=map_extents[3],
                    ax=ax)

        m.drawcoastlines(linewidth=2.0, zorder=zorder)
        m.drawcountries(linewidth=1.0, zorder=zorder)
        m.drawstates(linewidth=1.0, zorder=zorder)
        return ax, m

    def _truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = mplc.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    def contour_map(self, var, t_ind=0,
                    zero_to_nan=True,
                    cb=True, cb_label=None, cmap=None,
                    cb_min=None, cb_max=None, cb_nlev=201, cb_extend='neither', cb_log=False,
                    title=None,
                    ax=None):
        # Plot setup
        if ax is None:
            fig = pl.figure(figsize=(8 ,6))
            # pl.subplots_adjust(bottom=0.125, top=0.925, right=0.75)
            ax = pl.subplot(111)
        else:
            pl.sca(ax)
        ax, m = self._map_setup(ax, zorder=10)
        # Plot
        if cmap is None:
            cmap = self._truncate_colormap(pl.cm.jet, 0.1, 1.0, cb_nlev)
        if cb_max is None:
            cb_max = np.power(10, np.ceil(np.log10(np.nanmax(var))))
        if cb_min is None:
            if cb_log:
                if np.nanmin(var) < 0.:
                    raise Exception('Min value below zero. Log colorbar impossible.')
                else:
                    cb_min = 10 ** np.floor(np.log10(np.nanmin(var[var>0])))
            else:
                spam = 10 ** np.floor(np.log10(np.nanmin(var)))
                if spam < 0.:
                    cb_min = spam
                else:
                    cb_min = 0.
        pv = var[t_ind]
        if zero_to_nan:
            pv[pv==0.] = np.NaN
        if cb_log:
            lev_exp = np.arange(np.floor(np.log10(np.nanmin(var[var>0]))-1),
                               np.ceil(np.log10(np.nanmax(var))+1))
            norm = mplc.LogNorm()
            ticks = np.power(10, lev_exp)
            levs = np.logspace(np.log10(cb_min), np.log10(cb_max), cb_nlev)
            p = m.contourf(self.parent._data.lon[t_ind],
                       self.parent._data.lat[t_ind],
                       pv,
                       levels=levs, cmap=cmap, norm=norm,
                       latlon=True)
        else:
            levs = np.linspace(cb_min, cb_max, cb_nlev)
            ticks = np.linspace(cb_min, cb_max, 6)
            p = m.contourf(self.parent._data.lon[t_ind],
                       self.parent._data.lat[t_ind],
                       pv,
                       levels=levs, cmap=cmap,
                       extend=cb_extend,
                       latlon=True)
        if cb:
            if cb_max < 0.001:
                cb = m.colorbar(p, ticks=ticks, location='bottom', pad="10%", format='%.0e')
            else:
                cb = m.colorbar(p, ticks=ticks, location='bottom', pad="10%")
            if cb_label is not None:
                cb.set_label(cb_label)

        if title is not None:
            pl.title(title)

        # pl.show()

        return m

    def pop_AOD_example(self, output_dir, file_name):
        pl.figure(figsize=(10, 12))
        pl.subplots_adjust(bottom=0.07, top=0.97, left=0.07, right=0.97, hspace=0.35, wspace=0.3)
        nrow = 4
        ncol = 3
        pn = 0

        names = ['RAMS_salt_jet']
        wl_ad = '_550'
        aod_log = [False, False, False, False]
        aod_min = [None, None, None, None]
        aod_max = [0.10, 0.10, 1.0, 1.0]
        xlim = [1e1, 3e5]

        for i in range(len(names)):
            n = names[i]
            obj = getattr(self.parent.AOD, n)
            obj2 = getattr(self.parent._analysis, n+wl_ad)
            pn+=1
            ax = pl.subplot(nrow, ncol, pn)
            self.contour_map(obj.dry.AOD,
                             cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_extend='max', cb_label='AOD',
                             ax=ax)
            pn+=1
            ax = pl.subplot(nrow, ncol, pn)
            self.contour_map(obj.wet.AOD,
                             cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_extend='max', cb_label='AOD',
                             ax=ax)
            pn+=1
            if not obj2._mu_interp:
                ax = pl.subplot(nrow, ncol, pn)
                self.simple_dist_plot(obj2, 0, ax=ax, xlim=xlim)

        i = 3
        pn+=1
        ax = pl.subplot(nrow, ncol, pn)
        var = np.nansum([getattr(self.parent.AOD, n).dry.AOD for n in names], axis=0)
        self.contour_map(var,
                         cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_label='AOD', ax=ax)
        pn+=1
        ax = pl.subplot(nrow, ncol, pn)
        var = np.nansum([getattr(self.parent.AOD, n).wet.AOD for n in names], axis=0)
        self.contour_map(var,
                         cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_label='AOD', ax=ax)

        # pl.show()
        pl.savefig(os.path.join(output_dir, file_name + '-example.png'))
        pl.close()

    def pop_AOD_method1(self, output_dir, file_name):
        # - First plot with salt and dust -
        pl.figure(figsize=(10, 16))
        pl.subplots_adjust(bottom=0.07, top=0.97, left=0.07, right=0.97, hspace=0.35, wspace=0.3)
        nrow = 5
        ncol = 3
        pn = 0

        names = ['RAMS_salt_film', 'RAMS_salt_jet', 'RAMS_salt_spume', 'RAMS_dust1', 'RAMS_dust2']
        wl_ad = '_550'
        aod_log = [False, False, True, False, False]
        aod_min = [None, None, None, None, None]
        aod_max = [0.01, 0.10, 1e-5, 2.0, 2.0]
        xlim = [1e1, 3e5]

        for i in range(len(names)):
            n = names[i]
            obj = getattr(self.parent.AOD, n)
            obj2 = getattr(self.parent._analysis, n+wl_ad)
            pn+=1
            ax = pl.subplot(nrow, ncol, pn)
            self.contour_map(obj.dry.AOD,
                             cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_extend='max', cb_label='AOD',
                             ax=ax)
            pn+=1
            ax = pl.subplot(nrow, ncol, pn)
            self.contour_map(obj.wet.AOD,
                             cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_extend='max', cb_label='AOD',
                             ax=ax)
            pn+=1
            ax = pl.subplot(nrow, ncol, pn)
            self.simple_dist_plot(obj2, 0, ax=ax, xlim=xlim)

        # pl.show()
        pl.savefig(os.path.join(output_dir, file_name + '-1.png'))
        pl.close()

        # - Second plot with CCN, regen, and total
        pl.figure(figsize=(10, 14))
        pl.subplots_adjust(bottom=0.07, top=0.97, left=0.07, right=0.97, hspace=0.35, wspace=0.3)
        nrow = 4
        ncol = 3
        pn = 0

        names2 = ['RAMS_ccn', 'RAMS_regen_aero1', 'RAMS_regen_aero2']
        wl_ad = '_550'
        aod_log = [False, False, False, False, False]
        aod_min = [None, None, None, None, None]
        aod_max = [0.01, 0.10, 0.10, 0.10, 3.0]
        xlim = [1e1, 3e5]

        for i in range(len(names2)):
            n = names2[i]
            obj = getattr(self.parent.AOD, n)
            obj2 = getattr(self.parent._analysis, n + wl_ad)
            pn += 1
            ax = pl.subplot(nrow, ncol, pn)
            self.contour_map(obj.dry.AOD,
                             cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_extend='max', cb_label='AOD',
                             ax=ax)
            pn += 1
            ax = pl.subplot(nrow, ncol, pn)
            self.contour_map(obj.wet.AOD,
                             cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_extend='max', cb_label='AOD',
                             ax=ax)
            pn += 1
            ax = pl.subplot(nrow, ncol, pn)
            self.simple_dist_plot(obj2, 0, ax=ax, xlim=xlim)

        names_all = names + names2
        i = 4
        pn+=1
        ax = pl.subplot(nrow, ncol, pn)
        var = np.nansum([getattr(self.parent.AOD, n).dry.AOD for n in names_all], axis=0)
        self.contour_map(var,
                         cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_extend='max', cb_label='AOD',
                         ax=ax)
        pl.title('Total')
        pn+=1
        ax = pl.subplot(nrow, ncol, pn)
        var = np.nansum([getattr(self.parent.AOD, n).wet.AOD for n in names_all], axis=0)
        self.contour_map(var,
                         cb_log=aod_log[i], cb_min=aod_min[i], cb_max=aod_max[i], cb_extend='max', cb_label='AOD',
                         ax=ax)

        # pl.show()
        pl.savefig(os.path.join(output_dir, file_name + '-2.png'))
        pl.close()

    def pop_fRH(self):
        pl.figure(figsize=(8, 6))
        # pl.subplots_adjust(bottom=0.07, top=0.97, left=0.07, right=0.97, hspace=0.35, wspace=0.3)
        # nrow = 4
        # ncol = 3
        pn = 0

        wl_ad = '_550'

        # names = ['RAMS_salt_film', 'RAMS_salt_jet', 'RAMS_salt_spume']
        # names = ['RAMS_salt_film_alt', 'RAMS_salt_jet_alt', 'RAMS_salt_spume_alt']
        names = self.parent._pop_types
        # names = ['RAMS_salt_film', 'RAMS_salt_jet', 'RAMS_salt_spume',
        # #          'WRF_SEAS_1', 'WRF_SEAS_2', 'WRF_SEAS_3', 'WRF_SEAS_4']
        # colors = ['c','b','r',
        #           'c','g','b','r']
        # ls = ['-','-','-',
        #       '--','--','--','--']

        names = ['RAMS_salt_film', 'RAMS_salt_jet', 'RAMS_salt_spume',
                 'RAMS_dust1', 'RAMS_dust2',
                 'RAMS_ccn',
                 'RAMS_regen_aero1', 'RAMS_regen_aero2']
        colors = ['c','b','steelblue',
                  'DarkOrange', 'r',
                  'm',
                  'g','darkgreen']
        ls = ['-']*8

        for i in range(len(names)):
            n = names[i]
            # obj = getattr(self.parent.AOD, n)
            obj2 = getattr(self.parent._analysis, n+wl_ad)
            ax = pl.subplot(111)
            RH = obj2.opt.data.RH.d
            mass_ext_eff = obj2.opt.ext_cn(RH, obj2.opt._normMass2CNconc)
            p = pl.plot(RH, mass_ext_eff, label=n, c=colors[i], ls=ls[i])
            print(n, RH[0], mass_ext_eff[0])

        pl.yscale('log')
        pl.ylabel('Mass Ext Eff')
        pl.xlabel('RH')
        pl.legend()
        # pl.title('RAMS & WRF sea salt population types\nMass Extinction Efficiency - Testing')
        pl.title('RAMS population types\nMass Extinction Efficiency - Testing')
        pl.show()

        # Number distribution as well
        pl.figure(figsize=(8, 6))
        xlim = [1e1, 3e5]

        for i in range(len(names)):
            ax = pl.subplot(111)
            n = names[i]
            obj2 = getattr(self.parent._analysis, n+wl_ad)
            self.simple_dist_plot(obj2, 0, label=n, c=colors[i], ls=ls[i], dmy=i, ax=ax, xlim=None,
                                  plot_voldist=False)

        pl.xlim(xlim)
        # pl.xscale('log')
        pl.xlabel('Diameter (nm)')
        pl.ylabel('Normalized\ndN/dlogDp')
        pl.legend()
        # pl.title('RAMS & WRF sea salt population types\nNumber Size Distribution - Testing')
        pl.title('RAMS population types\nNumber Size Distribution - Testing')
        pl.show()

    def pop_plot_only1(self, CNdist, BlankObject, output_dir, file_name):
        # One plot for each pop type in file - hdf5 files only
        for pop_type in self.parent._pop_types:
            pl.figure(figsize=(10, 4))
            pl.subplots_adjust(bottom=0.15, top=0.85, left=0.07, right=0.97, hspace=0.35, wspace=0.3)
            nrow = 1
            ncol = 3
            pn = 0

            dry_AOD = np.array([self.parent._plot_f['wl_nm-550']['AOD']['dry'][pop_type].value])
            wet_AOD = np.array([self.parent._plot_f['wl_nm-550']['AOD']['wet'][pop_type].value])
            spam = pop_type + '_{}'.format(int(self.parent._wl))
            try:
                eggs = getattr(self.parent._analysis, spam)
            except:
                eggs = None
                obj2 = None
            if eggs:
                if eggs._mu_interp:
                    # HACK: only works in one situation as specified for case specific RAMS runs
                    ham = getattr(eggs, eggs.mu_dist_objs[0])._p_parms[0]
                    median_mu = np.median(getattr(self.parent._data, pop_type + '_medrad'))
                    ham[0] = median_mu
                    eggs = BlankObject()
                    eggs.n = 1
                    obj2 = CNdist(eggs, num_parm=ham, modes=1, CNconc=1., auto_bins=True, nbins=100, gen_dist=True)
                else:
                    obj2 = eggs

            def get_max(arr):
                v = np.nanmax(np.atleast_1d(arr))
                spam = np.power(10,np.ceil(np.log10(np.nanmax(v))))
                eggs = v/spam
                a = np.array([0.1,0.2,0.5,1.0])
                ham = a[np.where(eggs <= a)[0][0]]
                return min(spam*ham, 5.0)

            name = pop_type
            aod_log = False
            aod_min = None
            aodd_max = get_max(dry_AOD)
            aodw_max =get_max(wet_AOD)
            xlim = [1e1, 3e5]

            pn+=1
            ax = pl.subplot(nrow, ncol, pn)
            self.contour_map(dry_AOD,
                             cb_log=aod_log, cb_min=aod_min, cb_max=aodd_max, cb_extend='max', cb_label='AOD',
                             ax=ax, title='dry AOD')

            pn+=1
            ax = pl.subplot(nrow, ncol, pn)
            self.contour_map(wet_AOD,
                             cb_log=aod_log, cb_min=aod_min, cb_max=aodw_max, cb_extend='max', cb_label='AOD',
                             ax=ax, title='Humidified AOD')

            if pop_type != 'Total':
                pn += 1
                ax = pl.subplot(nrow, ncol, pn)
                self.simple_dist_plot(obj2, 0, ax=ax, xlim=xlim, title='Median Diameter: {:.2f} nm'.format(median_mu))

            fn_short = '.'.join(file_name.split('.')[:-1])
            pl.suptitle(fn_short)
            # pl.show()
            pl.savefig(os.path.join(output_dir, fn_short + '-simpleAOD.png'))
            pl.close()
