# csumuriaopt.py

"""
CSU MURI Aerosol Optical Analysis Code
A simplified set of python aerosol and optical analysis classes.
Created to support the CSU MURI.
Included here are classes for
(a) defining aerosol populations, including size distribution,
concentration, hygroscopicity, index of refraction, etc.,
(b) aerosol hygrscopic growth using kappa-Kohler theory,
(c) optical reconstructions using Mie theory.

The code is structured in an object oriented manner. Individual scenarios,
data points, or analyses can be created by instantiating a new instance of
one or more of the classes.

This set of code is intended primarily for parameterized aerosol populations.
Code for analyzing observed datasets with many data points, time or spatial
averaging, fitting and parameterizing methods, and machine learning methods
are not included here to simiplify use and analysis.
Email Sam Atwood if you would like any of these additional classes or methods.

Author: Sam Atwood, CSU, March 2017, satwood@atmos.colostate.edu

Dependencies:
This code utilizes the pymiecoated python package (Copyright 2012-2013
Jussi Leinonen), which implements Mie code based on Boren and Huffman, 1983.
The pymiecoated package has been slightly modified (as noted in the code)
to allow for changing the mie calculation memory cache size, and to correct
small bugs that were enountered. The modified version is included with this
package.

This code is intended to be run using Python 2. It is not fully compatible
with Python 3.

License:
MIT License

Copyright (c) 2017 Sam Atwood
pymiecoated License: Copyright (C) 2012-2013 Jussi Leinonen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# --- Imports ---
import sys
import os
import datetime as dt
import scipy as sp
import scipy.stats as sps
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import matplotlib.pylab as pl
#import matplotlib.dates as mdates
#import matplotlib.ticker as ticker
import pickle
#import dill as picklem
import copy
from numpy import intersect1d as i1
#from numpy import union1d as u1
import bisect
import fnmatch
import pymiecoated as pym
import gc
#import pdb


# --- Base Classes ---
"""
NOTE: This is an old methodology that was intended to support more advanced automatic data import, QA/QC screening,
and multi-dimensional averaging methods, along with automatic alignment of data between various types and sources
of observations. As a result, it is more complicated than needed for the purposes of this type of simplified
analysis. In addition, many of these classes and methods were built to learn how various python techniques work, and
could stand to be updated to make better use of more pythonic properties, more standard python procedures, or simply
replaced with a common python data analysis package like pandas. However, these existing analysis classes rely on
BaseAnalysis() and the associated methods. As a result, they still use these methods so that the analysis will
function without too much reworking of the code.
"""
class BlankObject(object):
    """A blank object for class composition.
    """
    def __init__(self, name=None):
        if name is not None:
            self.__name__ = name


class VarObject(object):
    """A blank variable object with standard data and information set as attributes.
    # TODO: Much of this is far too convoluted and should be similified and
    coverted to make use of more standard python property decorators.
    Attribute descriptions and types:
        d:              numpy array; data (can be 2-dimensional or higher for vector datapoints)
        v:              boolean numpy array; Validity of each datapoint
        typ:            numpy type object; the type of data in d
        meas:           boolean; True if averaging should be performed on this variable
        size:           int; number of datapoints in d
                            1: a constant scalar or vector for all data points
                            >1: a vector/scalar of length 'length' for each data point
                            None: for a value >1 that will be set later
        length:         int; length of each datapoint in d
                            1 for bool or scalar
                            2 or more for vector
        dim:            int; number of dimensions of numpy array in d
                            NOTE: vector data points have dim values of 2,
                            scalar data points have values of 1,
                            constant (at each data point / timestamp) vectors have values of 1,
                            and constant scalars have values of 0.
        vec:            numpy array or None; metadata values of higher order dimensions of d if vector data
                            NOTE: For now, this is a 1-D array of values associated with each vector
                            datapoint in d. This could be upgraded later if these values change with
                            each datapoint.
                            If vec is set to True, then will grab this 1-D array from the first datapoint
                            of another DataObject. (e.g. another DataObject holds bin sizes)
        units:          string (latex format); the units of the variable in latex format
        label_axis:     string; Label for an axis (without units)
        label_title:    string; Label that goes in title of plot
        desc:           string; General description of the variable
        circ:           True for circular data, False otherwise
        kwargs:         key for additional attribute names, value for associated values
    """
    def __init__(self, typ=float, meas=False, size=None, length=None, dim=1, vec=None, units='', label_axis='', label_title='', desc='',
                 circ=False,**kwargs):
        # set the attributes that describe this data time series
        self.d = np.array([])
        self.v = np.array([])
        self.typ = typ
        self.meas = meas
        self.size = size
        self.length = length
        self.dim = dim
        self.vec = vec
        self.units = units
        self.label_axis = label_axis
        self.label_title = label_title
        self.desc = desc
        self.circ = circ
        # Set any extra attributes that are passed
        #   Note: If d or v are passed, they should be arrays, and thus
        #   a copy of these are assigned to the attribute.
        for k,v in kwargs.items():
            if k == 'd':
                setattr(self,k,copy.copy(v))
            elif k == 'v':
                setattr(self,k,copy.copy(v))
            else:
                setattr(self,k,v)

class BaseAnalysis(object):
    """An inheritable data analysis support base class.
    """
    def __init__(self, name, desc, parent=None, obj_setpt=None):
        """
        Parameters:
            name:       Name of the analysis class or instance.
            desc:       Description of the analysis class or instance.
            parent:     The parent instance which this instance is composed within.
                        NOTE: This is a cheap recursion hack to make some composition functionality work better, but is not very pythonic.
            obj_setpt:  If not None, will set the self object at a different point than parent. This is primarily to allow
                        for an intermediate object to combine lots of similar AC instances together and keep them from
                        cluttering the data class namespace with lots of object attributes.
        """
        # Check for python 2
        # This code should mostly work with python 3, except for occasional print statements, but haven't verified yet
        if sys.version_info >= (3, 0):
            sys.stdout.write("This code has not yet been fully verified for Python 3.x")
            sys.exit(1)
        # Assign attributes
        self.parent = parent
        self.__name__ = name
        self.desc = desc
        # If parent is included, assign this Analysis Class Instance to the parent as an attribute
        # and add to parent.a analysis record unless overridden with incl_ac=False
        if parent is not None:
            if obj_setpt is None:
                setattr(parent, name, self)
            else:
                setattr(obj_setpt, name, self)

    def _make_var(self, varname, def_varname=None, data_obj=None, **kwargs):
        """Creates a VarObject instance that is filled with appropriate attributes.
        Note: 'varname' can contain spaces, but these will be removed in the attribute name.
        Arguments:
            varname:        The name of the variable being created
            def_varname:    If the variable dictionary name lookup is different from varname
            data_obj:       The data object for the a VarObject() variable. Default is self.data
            kwargs:         All additional key word arguments will override or add attributes to the variable
        """
        if data_obj is None:
            data_obj = self.data
        if def_varname is None:
            def_varname = varname
        varname = varname.replace(' ','').strip()
        pname = ''
        if hasattr(self.parent, '__name__'):
            if self.parent.__name__ is not None:
                pname = self.parent.__name__+' '
        args = self._var_dict(def_varname, dsname=pname)
        k = ('typ', 'meas', 'size', 'length', 'dim', 'vec', 'units', 'label_axis', 'label_title', 'desc')
        kw = {}
        for i in range(len(k)):
            kw[k[i]] = args[i]
        kw.update(kwargs)
        kw['parent'] = self.parent
        kw['name'] = varname
        kw['_def_varname'] = def_varname
        setattr(data_obj, varname, VarObject(**kw))

    def _set_var(self, varname, values, v=None, data_obj=None, vec=None, **kwargs):
        """Sets a variable's data (varname.d) and validity (varname.v) attributes from 'val' argument.
        Note: 'varname' can contain spaces, but these will be removed in the attribute name.
        If v is None, the varname.v attribute will be set to all True.
        data_obj: same behavior as self._make_var()
        vec is set to None (e.g. appropriate for a 1D variable) unless specified
        All kwargs will be set as additional attributes of data_obj.varname.kw_key=kw_value
        NOTE - Interpreting variable object dimensions: # TODO: bleh... should update and fix this... so bad.
            Examples:
            Vector variable - vector at each data point:
                dim: 2      size: 2     length: 3
            Vector constant - constant vector for all data points:
                dim: 1      size: 1     length: 3
            Scalar variable - scalar at each data point:
                dim: 1      size: 2     length: 1
            Scalar constant - constant scalar for all data points:
                dim: 0      size: 1     length: 1
        So if dim is 1, it could be either a vector constant or a scalar variable.
        For a vector constant, size must therefore be set to 1 when creating the
        variable with self._make_var or sometime before this method is called.
        The assumption with a dim of 1 is therefore that it is a scalar variable,
        unless otherwise specified by setting size equal to 1.
        In addition, constants (either scalar or vector) should have a .meas of False
        """
        if data_obj is None:
            data_obj = self.data
        varname = varname.replace(' ','').strip()
        # If the value passed is a variable object, set exactly and move to keywords
        if isinstance(values, VarObject):
            data_obj.__dict__[varname] = values
        # Otherwise, set attributes accordingly
        else:
            # Set .d
            data_obj.__dict__[varname].d = copy.copy(values)
            # Set .v
            if v is None:
                data_obj.__dict__[varname].v = np.ones(data_obj.__dict__[varname].d.shape, dtype=bool)
            elif np.isscalar(values) and np.isscalar(v):
                data_obj.__dict__[varname].v = copy.copy(v)
            else:
                if v.shape != data_obj.__dict__[varname].d.shape:
                    raise Exception
                data_obj.__dict__[varname].v = copy.copy(v)
            # Set .size, .length, and .vec if applicable
            if data_obj.__dict__[varname].dim == 0:
                # scalar constant
                if data_obj.__dict__[varname].length > 1:
                    raise Exception('0D variable object should have length of 1')
                if data_obj.__dict__[varname].size > 1:
                    raise Exception('0D variable object should have size of 1')
                if data_obj.__dict__[varname].meas:
                    raise Exception('scalar constant variable object should have meas of False')
                data_obj.__dict__[varname].size = 1
                data_obj.__dict__[varname].length = 1
            elif data_obj.__dict__[varname].dim == 1:
                if data_obj.__dict__[varname].size == 1:
                    # vector constant
                    if data_obj.__dict__[varname].length is None:
                        data_obj.__dict__[varname].length = data_obj.__dict__[varname].d.size
                    if data_obj.__dict__[varname].length != data_obj.__dict__[varname].d.size:
                        raise Exception('vector constant %s with length %i given, but value length is %i'
                                        %(varname, data_obj.__dict__[varname].length, data_obj.__dict__[varname].d.size))
                    if data_obj.__dict__[varname].meas:
                        raise Exception('vector constant variable object should have meas of False')
                    data_obj.__dict__[varname].vec = vec
                else:
                    # scalar variable
                    if data_obj.__dict__[varname].length is None:
                        data_obj.__dict__[varname].length = 1
                    if data_obj.__dict__[varname].length != 1:
                        raise Exception('scalar variable length should be 1, currently %i'%data_obj.__dict__[varname].length)
                    if data_obj.__dict__[varname].size is None:
                        data_obj.__dict__[varname].size = data_obj.__dict__[varname].d.size
                    if data_obj.__dict__[varname].size != data_obj.__dict__[varname].d.size:
                        raise Exception('scalar variable size of %i found, but value size is %i'
                                        %(data_obj.__dict__[varname].size, data_obj.__dict__[varname].d.size))
            elif data_obj.__dict__[varname].dim == 2:
                # vector variable
                data_obj.__dict__[varname].size = data_obj.__dict__[varname].d.shape[0]
                try:
                    data_obj.__dict__[varname].length = data_obj.__dict__[varname].d.shape[1]
                except IndexError:
                    raise Exception('variable object %s has dim of 2 but values do not have second dimension'%varname)
                data_obj.__dict__[varname].vec = vec
            else:
                raise Exception('Cannot currently handle dim larger than 2')
        # Set any additional keyword attributes
        for k,v in kwargs.items():
            setattr(data_obj.__dict__[varname], k, copy.copy(v))

    def _parent_var(self, varname, key, data_obj=None):
        """Creates a copy of a variable object with all attributes from parent.
        """
        if data_obj is None:
            data_obj = self.data
        varname = varname.replace(' ','').strip()
        setattr(data_obj, varname, VarObject())
        for k,v in self.parent.data.__dict__[key].__dict__.items():
            data_obj.__dict__[varname].__dict__[k] = copy.copy(v)

    def _var_vec(self, varname, data_obj=None):
        """Returns the appropriate vector for self.data.key.vec in this AnalysisClass.
        Returns False if variable name isn't found. #FIXME: Returning the None is still appropriate I think, but it curently insn't ever returning a False.
        Grabs the key of the variable that holds the vec values from self._var_dict with the globveckey flag set.
        Note that dataset_object._var_vec() is a method that must be included in any dataset wishing to send its vectors to other
        datasets for reduced dataset averaging as it is used by da.Alignment.align().
        """
        if data_obj is None:
            data_obj = self.data
        veckey = self._var_dict(varname, globveckey=True)
        if veckey is not None:
            return copy.copy(data_obj.__dict__[veckey].d)
        else:
            return None

    def _setup_dataobj(self, cdo, gvar={}, var={}):
        """Sets up and calculates standard calcs for a new data object attribute.
        gvar are global variables or metavariables that apply or are needed by
        standard variables in var.
        If self._defgvar or self._defvar exists, this method will also calculate
        all of these default variables that the class requires.
        self._def(g)var:    lists of strings with the variable names, which are all
                            associated with a calculation in self._default_calcs,
                            with self._defgvar being run first.
        Parameters:
            gvar:       Dictionary of global/meta variables being passed into the
                        analysis class instance.
                keys:   The name of each variable being set.
                values: Values to set variable key to with options:
                    string:             Will grab variable from self.parent.data.key.
                    numpy ndarray:      An array of values to set, with all .v set to True.
                    scalar:             A scalar value that will be turned into an array
                                        and set as the only value, with its .v set to True.
                    Variable Object:    A variable object to copy, with at minimum,
                                        .d and .v attributes.
                                        If .vec attribute is included, will override
                                        the .vec attribution returned by self._var_vec()
                                        in case the standard value returned is incorrect.
            var:        Same as gvar, but conducted after the variables in gvar.
        """
        # Set imported variables
        for varname, val in gvar.items():
            if isinstance(val, str):
                self._parent_var(varname, val, data_obj=cdo)
            elif isinstance(val, np.ndarray):
                self._make_var(varname, data_obj=cdo)
                self._set_var(varname, val, v=np.ones(val.shape, dtype=bool), data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
            elif np.isscalar(val):
                self._make_var(varname, data_obj=cdo)
                self._set_var(varname, val, v=True, data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
            else:
                self._make_var(varname, data_obj=cdo)
                if hasattr(val, 'vec'):
                    self._set_var(varname, val.d, v=val.v, data_obj=cdo, vec=val.vec)
                else:
                    self._set_var(varname, val.d, v=val.v, data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
        # Calculate any initial default global or meta variables not passed
        if hasattr(self, '_defgvar'):
            # Try once to see if needed vars are available for the calculation
            redo_list = []
            for varname in self._defgvar:
                if varname not in cdo.__dict__.keys():
                    rslt = self._default_calcs(varname, data_obj=cdo)
                    if rslt is False:
                        redo_list.append(varname)
                    else:
                        self._make_var(varname, data_obj=cdo)
                        self._set_var(varname, rslt[0], v=rslt[1], data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
            # Try again to see if needed vars are now available
            for varname in redo_list:
                if varname not in cdo.__dict__.keys():
                    rslt = self._default_calcs(varname, data_obj=cdo)
                    if rslt is False:
                        raise Exception('%s calculation failed'%varname)
                    else:
                        self._make_var(varname, data_obj=cdo)
                        self._set_var(varname, rslt[0], v=rslt[1], data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
        # Set standard imported variables that may depend on having the global/meta ones already imported
        for varname, val in var.items():
            if isinstance(val, str):
                self._parent_var(varname, val, data_obj=cdo)
            elif isinstance(val, np.ndarray):
                self._make_var(varname, data_obj=cdo)
                self._set_var(varname, val, v=np.ones(val.shape, dtype=bool), data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
            elif isinstance(val, VarObject):
                self._make_var(varname, data_obj=cdo)
                self._set_var(varname, val.d, v=val.v, data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
            elif np.isscalar(val):
                self._make_var(varname, data_obj=cdo)
                self._set_var(varname, np.array([val]), v=np.ones(1, dtype=bool), data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
            else:
                raise Exception('Passed variables must be strings, scalars, numpy arrays, or VarObject instances.')
        # Calculate any next default variables not passed
        if hasattr(self, '_defvar'):
            # Try once to see if needed vars are available for the calculation
            redo_list = []
            for varname in self._defvar:
                if varname not in cdo.__dict__.keys():
                    rslt = self._default_calcs(varname, data_obj=cdo)
                    if rslt is False:
                        redo_list.append(varname)
                    else:
                        self._make_var(varname, data_obj=cdo)
                        self._set_var(varname, rslt[0], v=rslt[1], data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))
            # Try again to see if needed vars are now available
            for varname in redo_list:
                if varname not in cdo.__dict__.keys():
                    rslt = self._default_calcs(varname, data_obj=cdo)
                    if rslt is False:
                        raise Exception('%s calculation failed'%varname)
                    else:
                        self._make_var(varname, data_obj=cdo)
                        self._set_var(varname, rslt[0], v=rslt[1], data_obj=cdo, vec=self._var_vec(varname, data_obj=cdo))

    def _var_dict(self, varname, dsname='', globveckey=False):
        """Returns VarObj info on all variables in this AnalysisClass.
        In 2D vector variable, the length of each datapoint is unknown and initially set to None.
        (typ, meas, size, length, dim, vec, units, label_axis, label_title, desc, kwargs)
        dsname: will add this to label_title and desc entries.
        This method also can return the global vector key to grab equivalent vectors from instances of this
        CNdist class in other dataset objects. If globveckey is True, will instead return the key to do this.
        """
        if varname == 'ts':
            if globveckey:
                return None
            return (dt.datetime,False,None,1,1,None,'', 'Time', 'Timeseries', 'Timestamp')
        elif varname == 'dN':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(\# cm^{-3})$','dN',dsname+'Bin Particle Count',dsname+'Bin Particle Count')
        elif varname == 'Norm_dN':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(\#)$','$d\widetilde{N}$',dsname+'Normalized dN',dsname+'Normalized dN')
        elif varname == 'Norm_dV':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(\mu m^3)$','$d\widetilde{V}$',dsname+'Normalized dV',dsname+'Normalized dV')
        elif varname == 'BinMid':
            if globveckey:
                return None
            return (float,False,1,None,1,None,'$(nm)$','$D_p$',dsname+'Bin Midpoint Diameter',dsname+'Bin Midpoint Diameter')
        elif varname == 'BinLow':
            if globveckey:
                return None
            return (float,False,1,None,1,None,'$(nm)$','$D_p$',dsname+'Bin Lower Edge Diameter',dsname+'Bin Lower Edge Diameter')
        elif varname == 'BinUp':
            if globveckey:
                return None
            return (float,False,1,None,1,None,'$(nm)$','$D_p$',dsname+'Bin Upper Edge Diameter',dsname+'Bin Upper Edge Diameter')
        elif varname == 'dlogDp':
            if globveckey:
                return None
            return (float,False,1,None,1,None,'$(-)$','$dlogD_p$',dsname+'Bin dlogDp',dsname+'Bin dlogDp')
        elif varname == 'dNdlogDp':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(cm^{-3})$','$dN/dlogD_p$',dsname+'dN/dlogDp',dsname+'dN/dlogDp')
        elif varname == 'Norm_dNdlogDp':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(cm^{-3})$','$d\widetilde{N}/dlogD_p$',dsname+'Normalized dN/dlogDp',dsname+'Normalized dN/dlogDp')
        elif varname == 'dSdlogDp':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(nm^2 cm^{-3})$','$dS/dlogD_p$',dsname+'dS/dlogDp',dsname+'dS/dlogDp')
        elif varname == 'dVdlogDp':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(\mu m^3 cm^{-3})$','$dV/dlogD_p$',dsname+'dV/dlogDp',dsname+'dV/dlogDp')
        elif varname == 'Norm_dVdlogDp':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(\mu m^3 cm^{-3})$','$d\widetilde{V}/dlogD_p$',dsname+'Normalized dV/dlogDp',dsname+'Normalized dV/dlogDp')
        elif varname == 'Norm_dbextdlogDp':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(cm^{-3})$','$d\widetilde{b_{ext}}/dlogD_p$',dsname+'Normalized dbext/dlogDp',dsname+'Normalized dbext/dlogDp')
        elif varname == 'dMdlogDp':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$()$','dM/dlogDp',dsname+'$dM/dlogD_p$',dsname+'dM/dlogDp')
        elif varname == 'mnf':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(-)$','Modal Number Fraction',dsname+'Modal Number Fraction',dsname+'Modal Number Fraction')
        elif varname == 'mvf':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(-)$','Modal Volume Fraction',dsname+'Modal Volume Fraction',dsname+'Modal Volume Fraction')
        elif varname == 'Dbar':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(nm)$','Number Mean Diameter',dsname+'Number Mean Diameter',dsname+'Number Mean Diameter')
        elif varname == 'DVbar':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(nm)$','Volume Mean Diameter',dsname+'Volume Mean Diameter',dsname+'Volume Mean Diameter')
        elif varname == 'CNconc':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(\#/cm^3)$','CN Concentration',dsname+'Reconstructed CN Concentration',dsname+'Reconstructed CN Concentration')
        elif varname == 'Volconc':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(\mu m^3/cm^3)$','Volume Concentration',dsname+'Reconstructed Volume Concentration',dsname+'Reconstructed Volume Concentration')
        elif varname == 'ReconSca':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(Mm^{-1})$','Scattering Coef',dsname+'Reconstructed Scattering Coefficient',dsname+'Reconstructed Scattering Coefficient')
        elif varname == 'Recon_dSca':
            if globveckey:
                return None
            return (float,True,None,None,2,None,'$(Mm^{-1} cm^{-3})$','$db_{sca}/dlogD_p$',dsname+'Reconstructed Scattering Coefficient Distribution $db_{sca}/dlogD_p$',dsname+'Reconstructed Scattering Coefficient Distribution $db_{sca}/dlogD_p$')
        elif varname == 'Height':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(m)$','Model Height',dsname+'Model Height',dsname+'Model Height')
        #elif varname == 'RH':
        #    if globveckey:
        #        return None
        #    return (float,True,None,None,2,None,'$(\%)$','Model RH',dsname+'Model RH',dsname+'Model RH')
        elif varname == 'RH':
            if globveckey:
                return None
            return (float,True,None,None,1,None,'$(\%)$','Model RH',dsname+'Model RH',dsname+'Model RH')
        elif varname == 'dry_m':
            if globveckey:
                return None
            return (float,True,None,None,2,None,'$(-)$','dry m',dsname+'Dry index of refraction',dsname+'Dry complex index of refraction')
        elif varname == 'kappa':
            if globveckey:
                return None
            return (float,True,None,None,2,None,'$(-)$','kappa',dsname+'Kappa',dsname+'Modal Kappa Hygroscopicity Parameter')
        elif varname == 'rhosol':
            if globveckey:
                return None
            return (float,True,None,None,2,None,'$(g/cm^3)$','rhosol',dsname+'Solute Density',dsname+'Modal Solute Density')
        else:
            print 'Warning: varname "%s" not found in _var_dict()'%varname
            return None


class Dist(object):
    """An inheritable class with distribution function objects.
    NOTE: Only typical lognormal with variable amplitude (i.e. number concentration) included here.
    """
    def lognormal_pdfv(self, x, *parms, **kwargs):
        """Returns the probability density function value for lognormal distribution with an amplitude.
        Note that this function is not invertible as it does not monotonically increase.
        Parameters:
            x:      Value where f(x) is to be returned
            *parms:
                median: Median of distribution
                sigma:  Geometric Standard Deviation
                a:      Amplitude multiplier
        Source: Seinfeld and Pandis 2nd ed. section 8.1.6.
        """
        if len(parms) != 3: raise Exception(parms)
        median, sigma, a = parms
        if 'inv' in kwargs:
            if kwargs['inv']:
                print 'Warning: A lognormal density function is not invertable, NaN returned'
                return np.NaN
        return ( (a/(np.log10(sigma)*np.sqrt(2.*np.pi))) * np.exp((np.log10(x)-np.log10(median))**2/(-2.*(np.log10(sigma))**2)) )

    def _mix_dist_f(self, x, func, modes, pm, *parms, **kwargs):
        '''The forward calc for self.mix_dist.
        '''
        p = np.reshape(parms,[modes,pm])
        yarr = np.zeros([modes, np.size(x)])
        y = np.array(0)
        for n in range(modes):
            yarr[n] = func(x, *p[n])
            y = y + yarr[n]
        if 'fr' in kwargs:
            if kwargs['fr']:
                return (y, yarr)
        return y

    def _mix_dist_i(self, y, func, modes, pm, bounds=None, *parms, **kwargs):
        '''The inverse calc for self.mix_dist.
        '''
        if 'init' in kwargs:
            init = kwargs['init']
        else:
            init = 0.1
        if 'dif_tol' in kwargs:
            dif_tol = kwargs['dif_tol']
        else:
            dif_tol = 1e-8
        # Assume the result is invalid until evaluated to be valid
        f = lambda z: y-self._mix_dist_f(z, func, modes, pm, *parms)
        # if func is bounded in x, use brentq, otherwise use root
        if bounds:
            if len(bounds) != 2:
                raise Exception
            x0, r = sp.optimize.brentq(f, bounds[0], bounds[1], full_output=True, disp=False)
            if r.converged:
                return x0
        else:
            xsolve = sp.optimize.root(f, init)
            if xsolve.success:
                if xsolve.fun < dif_tol:
                    return xsolve.x
            # If that didn't work, try another root method
            xsolve = sp.optimize.root(f, init, method='lm')
            if xsolve.success:
                if xsolve.fun < dif_tol:
                    return xsolve.x
        return np.NaN

    def mix_dist(self, x, func, parms, modes=1, inv=False, neg=False,
                 dif_tol=1e-8, max_result=None, bounds=None, fr=False):
        '''Returns the expected value at x of f(x), a mixture distribution.
        Parameters:
            x:          Values where f(x) is to be returned.
            func:       The function which returns an expected value for a single mode distribution.
            parms:      1D List of parameters. Should have #parms per mode * #modes
        Optional Keyword Parameters:
            modes:      The number of modes of func the combined distribution should have.
            inv:        If True, will instead return the x at f(x). Will fail if func is not
                        a monotonically increasing function, or appropriate xbounds in x are passed.
                dif_tol:    The maximum allowable difference between the return value and f(x)
                max_result: The maximum allowable return value, f(x) if inv is called.
            neg:        If True, returns result * -1. Does nothing if inv is True.
            bounds:     If not None, any parameters passed to this function with parameters
                        outside of the bounds will return NaN.
                        Format is [(min,max),...] tuples for each parameters in mixture distribution.
            fr:         If True, full return of (f(x) value, individual modal y values). Only for forward dist.
        '''
        # Return NaN if input parms is NaN
        if parms is np.NaN:
            return np.NaN
        # Quick check to make sure parameters and modes were sent correctly
        if np.mod(len(parms),modes) != 0.:
            raise Exception('mix_dist Error: Parameters: %i, Modes: %i'%(len(parms),modes))
        pm = len(parms)/modes    #parameters per mode
        # If bounds is set, do bounds check
        if inv is False and bounds is not None:
            if len(bounds) != len(parms):
                raise Exception('mix_dist Error: Parameter: %i, Bounds: %i'%(len(parms),len(bounds)))
            for i in range(len(parms)):
                if parms[i] < bounds[i][0] or parms[i] > bounds[i][1]:
                    return np.NaN
        # If the inverse is requested
        if inv:
            if max_result is not None:
                if x > max_result:
                    return np.NaN
            return self._mix_dist_i(x, func, modes, pm, *parms, dif_tol=dif_tol, bounds=bounds)
        else:
            if neg:
                return -1. * self._mix_dist_f(x, func, modes, pm, *parms, fr=fr)
            return self._mix_dist_f(x, func, modes, pm, *parms, fr=fr)


class DistFit(object):
    """An inheritable class with statistics methods, primarily for distribution stats.
    Note that all private methods in this class do not have error checking on inputs.
    Caution should be exercised when using these methods to ensure that the methods
    are called correctly with the proper argument types and shapes.
    NOTE: Only typical lognormal with variable amplitude (i.e. number concentration) included here.
          No distribution fitting methods or model selection criteria methods are included here.
    """
    def __init__(self):
        """Composes required external classes rather than using inheritance.
        """
        self.dist = Dist()

    def _fitlib(self, func, **kwargs):
        """Returns the parameters and information for each fit distribution.
        Return tuple:
            f:              A python method object of the function to be used
            d:              Description of the distribution
            n_parms:        The number of parameters in each modal function
            parm_names:     The name of each parameter for each modal function
            parm_desc:      Associated descriptions of each
        """
        if func == 'LognormPDFv':
            '''A lognormal dN_dlogDp density function distribution.
            '''
            f = self.dist.lognormal_pdfv
            n_parms = 3
            d = 'Lognormal 3 parameter distribution'
            parm_names = ['median','sigma','N']
            parm_desc = [' - Modal Median Parameter',
                         ' - Modal Geometric Standard Deviation Parameter',
                         ' - Modal Reconstructed Number Count Parameter']
            return f, d, n_parms, parm_names, parm_desc


# --- Analysis Classes ---
class CNdist(BaseAnalysis):
    """A simplified CN distribution analysis and parameterization class.

    To allow for separate modes to have their own variables (that sum to form the variables in
    self.data), modes of the distribution can be stored as m# attributes of self with their own
    data object (e.g. self.m1.data.dNdlogDp). self._defvarlist is not implemented for these modes
    as (for now) all of these variables are the same across all similar data objects of each mode
    and the total distribution; self._defvarlist_. #TODO: This was quickly thrown together, and
    is still a bit wonky. Should probably be updated and standardized a bit better.

    Observed CN distributions:
        NOTE: This version of the code is not intended to handle variable observed distributions
              in order to simplify use of this code. Using the parameterization options is
              recommended.
    Any variables to be passed from the parent to the analysis class instance are passed in
    the 'var' argument, a dictionary of 'key':parent.data.'value'.d pairs.
    Standard Variables are required to be passed to this class, optional ones are automatically
    calculated if they aren't passed. Note that optional variables must be calculated in the
    order listed here.
    Standard Variables:
        dN:             Raw counts of particles (#/cm^3)
        BinLow:         Lower bin edge diameter for each dN (nm)
        BinUp:          Upper bin edge diameter for each dN (nm)
      Alternately to BinLow and BinUp, if bins assumed touching:
        BinMid:         Midpoint bin diameter for each dN (nm)
    Optional Variables:
        dlogDp:         log10(BinUp/BinLow)
        dNdlogDp:       Number (#/cm^3 * nm^-1)
        dSdlogDp:***#TODO:       Surface Area ()
        dVdlogDp:       Volume (#/cm^3 * nm^2)
        dMdlogDp:***#TODO:       Mass ()
        reconstructed number, SA, volume, mass should be here as well #TODO:

    Parameterized aerosol distributions:
    Alternately, aerosol distribution parameters can be passed to this class to define the
    distributions rather than using observed data. The same variables described above are
    calculated, but only distribution parameters are required to be passed an instantiation.
    See the _param_dist() and __init__() method docstrings for more information.

    Class Attributes:
        ndp:            Number of data points in this CNdist instance
        nmodes:         If a parameterized CNdist, the number of modes, otherwise None

    Class Methods:
      These can be run at instantiation by calling the method name and arguments as a
      dictionary entry in the 'calc' argument in __init__, or at any further time.
      See each method's docstring for more info on usage and arguments.
        combine_bins:       Combines bins into a fewer number
        num/vol_mean:
        num/vol_recon:
        generate_bins:      Automatically create bins to span a distribution
        _param_dist:        Create a distribution from parameterized values
        norm_dN/VdlogDp:    Create a normalized distribution variable
        bin_volume/mass:
        touching_bins:      Create touching bin edges from midpoints
        NOTE: none of the distribution fitting methods included in this version
        (e.g. multi-modal lognormal MLE, etc.) to simplify use and analysis with this code
    """
    def __init__(self, parent, gvar={}, var={}, calc=None, name_ovr=None, obj_setpt=None,
                 gen_dist=False, num_parm=None, vol_parm=None, CNconc=None, Volconc=None, auto_bins=False, nbins=200,
                 modes=None, modal_model='LognormPDFv'):
        """
        Arguments:
            parent:     The parent of the CNdist() instance.
            gvar:       Dictionary of global/meta variables being passed into the analysis class instance.
                        The value of each key can be a string to grab from parent.data.key,
                        or a numpy array to hold the actual values themselves.
            var:        Standard variables to be imporeted.
                        Same as gvar, but conducted after the variables in gvar.
            calc:       An (optionally ordered) dictionary of any requested calculation methods to run on instantiation.
            name_ovr:   Will override the name of the analysis class instance. Should typically be used only when
                        creating a second or non-typical version of this type of distribution analysis.
            obj_setpt:  Allows for the instance to be set at a different point from the dataset parent.
                        Typically, for adding an interim object to store multiple instances together in
                        a unique namespace (e.g. All CNdist() instances found in self.cn_dists.__dict__).
            gen_dist:   If True, will generate the size distributions from a parameterization.
                        The parameters parms and CNconc can be pulled from a variable being impored
                        in gvar or var by passing a string (self.data.__dict__[string]) for a variable
                        that holds the appropriate values for the parameter.
                        Variables for the total distribution will be stored normally, while any modes (if more than one)
                        will be stored with the same variables as .m# mode attributes, each with their own data objects.
                num_parm:       A 1D list of number distribution parameters to keep the same parameters for all datapoints, or
                                a 2D list of number distribution parameters of size self.data.n for different parameters
                                for each datapoint.
                vol_parm:       A 1D list of volume distribution parameters to keep the same parameters for all datapoints, or
                                a 2D list of volume distribution parameters of size self.data.n for different parameters
                                for each datapoint.
                CNconc:         The total number concentration of the population, or a 1D list of number concentrations
                                for each datapoint.
                Volconc:        The total volume concentration of the population, or a 1D list of volume concentrations
                                for each datapoint.
                modes:          The maximum number of modes in the distribution to be generated.
                modal_model:    The modal model, typically lognormal, to generate the distributions from.
                auto_bins:      If True, will create a new set of lognormally spaced initial bins that
                                capture most of size distribution. Low end is defined as the diameter
                                where dN/dlogDp = 0.001 * max(dN/dlogDp) and the upper end
                                where dV/dlogDp = 0.001 * max(dV/dlogDp). If BinMid, or BinLow and BinUp, are
                                not given, will set this to True.
                nbins:          The number of bins to use if auto_bins is True.
        """
        # Name and Description
        name = 'cn'
        desc = 'CN Size Distribution Analysis'
        if name_ovr is not None:
            name = name_ovr
        # Initialize class attributes
        self.ndp = None
        self.nmodes = None
        # Needed analysis classes
        self.dfit = DistFit()
        # Call BaseAnalysis Class init method to finish import
        BaseAnalysis.__init__(self, name, desc, parent, obj_setpt=obj_setpt)
        # Determine if generation of bin sizes is needed
        if gen_dist:
            if not ( 'BinMid' in gvar.keys() or ('BinLow' in gvar.keys() and 'BinUp' in gvar.keys()) ):
                auto_bins = True
            #TODO: Note that auto_bins will cause problems with more than one datapoint, as it will try
            #       to create a new data.BinMid variable customised to each datapoint. Haven't updated
            #       the code to easily allow for different BinMids in the same dataset so far.
            if auto_bins and self.parent.n > 1:
                raise Exception('The auto_bins method causes problem when there is more than one datapoint.')
        # List of default calculated variables all instances of this class will have
        self._defgvar = ['BinLow','BinUp','BinMid','dlogDp']
        #   For parameterizing, wait to calculate until after param is run
        if gen_dist:
            self._defvar = []
            if auto_bins:
                self._defgvar = []
        #   If distribution variables are given (not generating from parameters)
        else:
            self._defvar = ['dN','dNdlogDp','dVdlogDp']
        # Scrub certain types of input variables by proper dimension
        if CNconc is not None:
            CNconc = np.atleast_1d(CNconc)
        if Volconc is not None:
            Volconc = np.atleast_1d(Volconc)
        if 'kappa' in var:
            var['kappa'] = np.atleast_2d(var['kappa'])
        if 'dry_m' in var:
            var['dry_m'] = np.atleast_2d(var['dry_m'])
        # Create default data attribute and setup variables
        setattr(self, 'data', BlankObject())
        self._setup_dataobj(self.data, gvar, var)
        # Set number of data points in CNdist instance
        if CNconc is not None:
            self.ndp = CNconc.size
        elif Volconc is not None:
            self.ndp = Volconc.size
        # Generate size distributions from a parameterized if needed
        if gen_dist:
            self._param_dist(num_parm, vol_parm, CNconc, Volconc, modes, modal_model, auto_bins=auto_bins, nbins=nbins)
        # Run any additional requested calculations
        if calc is not None:
            for calc_name, calc_args in calc.items():
                if calc_name is 'combine_bins':
                    self.combine_bins(*calc_args)
                elif calc_name is 'num_mean':
                    self.num_mean(**calc_args)
                elif calc_name is 'vol_mean':
                    self.vol_mean(**calc_args)
                elif calc_name is 'num_recon':
                    self.num_recon(**calc_args)
                elif calc_name is 'vol_recon':
                    self.vol_recon(**calc_args)
                elif calc_name is 'norm_dNdlogDp':
                    self.norm_dNdlogDp(**calc_args)
                elif calc_name is 'norm_dVdlogDp':
                    self.norm_dVdlogDp(**calc_args)
                else:
                    raise Exception('Calculation name not found')

    def _var_dict(self, varname, dsname='', globveckey=False):
        """Returns VarObj info on all variables in this AnalysisClass.
        In 2D vector variable, the length of each datapoint is unknown and initially set to None.
        (typ, meas, size, length, dim, vec, units, label_axis, label_title, desc, kwargs)
        dsname: will add this to label_title and desc entries.
        This method also can return the global vector key to grab equivalent vectors from instances of this
        CNdist class in other dataset objects. If globveckey is True, will instead return the key to do this.

        If not found here, will look in BaseAnalysis._var_dict.

        Only variables specific to this class only (e.g. very generic variables like 'location', that have
        specific format requirements) should be listed here. Otherwise, they are safe to add in the
        BaseAnalysis variable dictionary (e.g. they are applicable to multiple classes).
        """
        if varname == '':
            if globveckey:
                return None
            return (bool,False,None,1,1,None,'', '', '', '')
        else:
            # If not local, look in Base Analysis
            return super(CNdist, self)._var_dict(varname, dsname=dsname, globveckey=globveckey)

    def _default_calcs(self, varname, mode=None, data_obj=None):
        """Calculates a variable's data (varname.d) and validity (varname.v) attributes.
        As these are all calculated variables, the validity is determined from the variables it uses
        to perform the calculations.
        data_obj:   The data object for the a VarObject() variable. Default is self.data
        If the required other variables are not available, will return False.
        If successful, returns a tuple of (value, validity) arrays.

        the mode parameter, if specified, contains a string with the modal suffix for the variable to be used.
        """
        if data_obj is None:
            data_obj = self.data
        avail = data_obj.__dict__.keys()
        # Calculate the data values
        if varname is 'dN':
            # Requires dNdlogDp and dlogDp
            if 'dNdlogDp' not in avail or 'dlogDp' not in avail:
                return False
            val = data_obj.dNdlogDp.d * data_obj.dlogDp.d
            v = data_obj.dNdlogDp.v
        elif varname is 'BinMid':
            # Requires BinLow and BinUp
            if 'BinLow' not in avail or 'BinUp' not in avail:
                return False
            val = np.sqrt(data_obj.BinUp.d*data_obj.BinLow.d)
            v = data_obj.BinUp.v & data_obj.BinLow.v
        elif varname is 'BinLow':
            # Requires BinMid and assumes touching bins
            if 'BinMid' not in avail:
                return False
            val, dmy = self.touching_bins(data_obj.BinMid.d)
            v = data_obj.BinMid.v
        elif varname is 'BinUp':
            # Requires BinMid and assumes touching bins
            if 'BinMid' not in avail:
                return False
            dmy, val = self.touching_bins(data_obj.BinMid.d)
            v = data_obj.BinMid.v
        elif varname is 'dlogDp':
            # Requires BinLow and BinUp
            if 'BinLow' not in avail or 'BinUp' not in avail:
                return False
            val = np.log10(data_obj.BinUp.d/data_obj.BinLow.d)
            v = data_obj.BinUp.v & data_obj.BinLow.v
        elif varname is 'dlnDp':
            # Requires BinLow and BinUp
            if 'BinLow' not in avail or 'BinUp' not in avail:
                return False
            val = np.log(data_obj.BinUp.d/data_obj.BinLow.d)
            v = data_obj.BinUp.v & data_obj.BinLow.v
        elif varname is 'dNdlogDp':
            # Requires dN and dlogDp OR dVdlogDp and BinMid
            if 'dN' not in avail or 'dlogDp' not in avail:
                if 'dVdlogDp' not in avail or 'BinMid' not in avail:
                    return False
                # Includes a unit conversion from um^3 to nm^3
                val = data_obj.dVdlogDp.d / (np.pi/6.) / (data_obj.BinMid.d)**3 / 1e-9
                v = data_obj.dVdlogDp.v
            else:
                val = data_obj.dN.d / data_obj.dlogDp.d
                v = data_obj.dN.v
        elif varname is 'dVdlogDp':
            # Requires dNdlogDp and BinMid
            if 'dNdlogDp' not in avail or 'BinMid' not in avail:
                return False
            # Includes a unit conversion from nm^3 to um^3
            val = data_obj.dNdlogDp.d * (np.pi/6.) * (data_obj.BinMid.d)**3 * 1e-9
            v = data_obj.dNdlogDp.v
        return (val, v)

    def combine_bins(self, name, bin_ind):
        """Combines raw counts (self.data.dN) bins by bin indicies and saves in new data object.
        The bin edges are grabbed from the self.data.BinUp and BinLow variable objects for each new bin.
        Arguments:
            name:       The name of the new data object that is assigned as an attribute of self
            bin_ind:    A dictionary that assigns old bin indicies to new bin indicies as
                            keys: the new bin indicies
                            values: list of the old bin indices in the new bin index
        """
        # Create a new analysis instance and data objects within this analysis instance to hold the new bin analysis
        setattr(self, name, BlankObject(name))
        setattr(self.__dict__[name], 'data', BlankObject())
        # Create and combine new dN, BinLow, and BinUp variable objects
        #   By using the for loop, it will also ensure that each bin index passed in bin_ind
        new_dN = []
        new_BinLow = []
        new_BinUp = []
        new_nbins = len(bin_ind)
        for i in range(new_nbins):
            new_dN.append(np.sum(self.data.dN.d[:,bin_ind[i]], axis=1))
            new_BinLow.append(np.min(self.data.BinLow.d[bin_ind[i]]))
            new_BinUp.append(np.max(self.data.BinUp.d[bin_ind[i]]))
        # Add BinLow and BinUp variables
        self._make_var('BinLow', data_obj=self.__dict__[name].data)
        self._set_var('BinLow', np.array(new_BinLow), data_obj=self.__dict__[name].data)
        self._make_var('BinUp', data_obj=self.__dict__[name].data)
        self._set_var('BinUp', np.array(new_BinUp), data_obj=self.__dict__[name].data)
        # Calculate BinMid variable
        self._make_var('BinMid', data_obj=self.__dict__[name].data)
        rslt = self._default_calcs('BinMid', data_obj=self.__dict__[name].data)
        self._set_var('BinMid', rslt, data_obj=self.__dict__[name].data)
        # Set dN now that vector is available
        self._make_var('dN', data_obj=self.__dict__[name].data)
        self._set_var('dN', np.array(new_dN).transpose(), data_obj=self.__dict__[name].data, vec=self.__dict__[name].data.BinMid.d)
        # Run _setup_dataobj to finish with any other calcs in self._defvar
        self._setup_dataobj(self.__dict__[name].data)

    def num_mean(self, varname='Dbar', v=None, data_obj=None, low=None, up=None, bin_ind=True):
        """Calculates the Number Mean Diameter of size distributions between low and up bin midpoint sizes.
        Uses data_obj.dN and data_obj.BinMid variables for calculation. (Seinfeld and Pandis, 2nd Ed., pg. 361).
        if bin_ind is True, low in the lowest bin index and up is the largest bin index included in the calculation
        If bin_ind is False, will find the lowest bin size greater than low for the lower index, and similar for up.
        """
        if data_obj is None:
            data_obj = self.data
        if v is None:
            # Assume all true
            v = np.ones(self.ndp, dtype=bool)
        if bin_ind:
            if low is None:
                low = 0
            if up is None:
                up = data_obj.BinMid.length - 1
            val_ind = range(low,up+1)
        else:
            if low is None:
                low = data_obj.BinMid.d[0]
            if up is None:
                up = data_obj.BinMid.d[-1]
            val_ind = np.where( (data_obj.BinMid.d >= low) & (data_obj.BinMid.d <= up) )[0]
        val = []
        for i in range(self.ndp):
            vind = i1(np.where(data_obj.dN.v[i])[0], val_ind)
            # all bins must be valid for a valid Dbar value
            if vind.size < len(val_ind) or not v[i]:
                val.append(np.NaN)
                v[i] = False
            else:
                val.append(np.sum(data_obj.dN.d[i,vind]*data_obj.BinMid.d[vind]) / np.sum(data_obj.dN.d[i,vind]))
        val = np.array(val)
        self._make_var(varname, def_varname='Dbar', data_obj=data_obj)
        self._set_var(varname, val, v=v, data_obj=data_obj)

    def vol_mean(self, varname='DVbar', v=None, data_obj=None, low=None, up=None, bin_ind=True):
        """Calculates the Volume Mean Diameter of size distributions between low and up bin midpoint sizes.
        Uses self.dVdlogDp, self.BinMid, and self.dlogDp variables for calculation. (Seinfeld and Pandis, 2nd Ed., pg. 361).
        If bin_ind is False, will find the lowest bin size included low for the lower index, and similar for up.
        NOTE: The units of dV/dlogDp are um^-3 cm^-3 for ease of use, but BinMid is in nm. So in order to calculate this
        property, a unit conversion on dVdlogDp is first performed.
        """
        if data_obj is None:
            data_obj = self.data
        if v is None:
            # Assume all true
            v = np.ones(self.ndp, dtype=bool)
        if bin_ind:
            if low is None:
                low = 0
            if up is None:
                up = data_obj.BinMid.length - 1
            val_ind = range(low,up+1)
        else:
            if low is None:
                low = data_obj.BinMid.d[0]
            if up is None:
                up = data_obj.BinMid.d[-1]
            val_ind = np.where( (data_obj.BinMid.d >= low) & (data_obj.BinMid.d <= up) )[0]
        val = []
        for i in range(self.ndp):
            vind = reduce(i1, (np.where(data_obj.dN.v[i])[0], np.where(data_obj.dVdlogDp.v[i])[0], val_ind))
            # all bins must be valid for a valid Dbar value
            if vind.size < len(val_ind) or not v[i]:
                val.append(np.NaN)
                v[i] = False
            else:
                val.append( ( np.sum(data_obj.dVdlogDp.d[i,vind]*1e9 * data_obj.dlogDp.d[vind]) / np.sum(data_obj.dN.d[i,vind]) * (6./np.pi) )**(1./3.) )
        val = np.array(val)
        self._make_var(varname, def_varname='DVbar', data_obj=data_obj)
        self._set_var(varname, val, v=v, data_obj=data_obj)

    def num_recon(self, varname='CNconc', v=None, data_obj=None, altdNkey=None, low=None, up=None, bin_ind=True):
        """Calculates the reconstructed number count of size distributions between low and up bin midpoint sizes.
        Uses self.dNdlogDp, self.dlogDp, and self.BinMid variables for calculation.
        if altdNkey is given, should hold a string with the key for an alternate data_obj.dNdlogDp variable.
        If bin_ind is False, will find the lowest bin size included low for the lower index, and similar for up.
        """
        if data_obj is None:
            data_obj = self.data
        if altdNkey is None:
            dNdlogDp = data_obj.dNdlogDp
        else:
            dNdlogDp = data_obj.__dict__[altdNkey]
        if v is None:
            # Assume all true
            v = np.ones(self.ndp, dtype=bool)
        if bin_ind:
            if low is None:
                low = 0
            if up is None:
                up = data_obj.BinMid.length - 1
            val_ind = range(low,up+1)
        else:
            if low is None:
                low = data_obj.BinMid.d[0]
            if up is None:
                up = data_obj.BinMid.d[-1]
            val_ind = np.where( (data_obj.BinMid.d >= low) & (data_obj.BinMid.d <= up) )[0]
        val = []
        for i in range(dNdlogDp.d.shape[0]):
            vind = i1(np.where(dNdlogDp.v[i])[0], val_ind)
            if vind.size > 0:
                val.append(np.sum(dNdlogDp.d[i,vind]*data_obj.dlogDp.d[vind]))
            else:
                val.append(np.NaN)
            if (~dNdlogDp.v[i]).all():
                v[i] = False
        val = np.array(val)
        self._make_var(varname, def_varname='CNconc', data_obj=data_obj)
        self._set_var(varname, val, v=v, data_obj=data_obj)

    def vol_recon(self, varname='Volconc', v=None, data_obj=None, altdVkey=None, low=None, up=None, bin_ind=True):
        """Calculates the reconstructed volume concentration of size distributions between low and up bin midpoint sizes.
        Uses self.dVdlogDp, self.dlogDp, and self.BinMid variables for calculation.
        If bin_ind is False, will find the lowest bin size included low for the lower index, and similar for up.
        """
        if data_obj is None:
            data_obj = self.data
        if altdVkey is None:
            dVdlogDp = data_obj.dVdlogDp
        else:
            dVdlogDp = data_obj.__dict__[altdVkey]
        if v is None:
            # Assume all true
            v = np.ones(self.ndp, dtype=bool)
        if bin_ind:
            if low is None:
                low = 0
            if up is None:
                up = data_obj.BinMid.length - 1
            val_ind = range(low,up+1)
        else:
            if low is None:
                low = data_obj.BinMid.d[0]
            if up is None:
                up = data_obj.BinMid.d[-1]
            val_ind = np.where( (data_obj.BinMid.d >= low) & (data_obj.BinMid.d <= up) )[0]
        val = []
        for i in range(dVdlogDp.d.shape[0]):
            vind = i1(np.where(dVdlogDp.v[i])[0], val_ind)
            if vind.size > 0:
                val.append(np.sum(dVdlogDp.d[i,vind]*data_obj.dlogDp.d[vind]))
            else:
                val.append(np.NaN)
            if (~dVdlogDp.v[i]).all():
                v[i] = False
        val = np.array(val)
        self._make_var(varname, def_varname='Volconc', data_obj=data_obj)
        self._set_var(varname, val, v=v, data_obj=data_obj)

    def generate_bins(self, num_parm=None, vol_parm=None,
                      modes=1, modal_model='LognormPDFv',
                      nbins=200, range_min=1, range_max=10000,
                      frac=.999):
        """Generates a bin array based on the edges of a given size distribution parameterization.
        Edges are generated based on expanding the bins until 99.9% (default) or greater of the number and
        volume distributions are captured.
        """
        if not (frac < 1 and frac > 0):
            raise Exception('frac must be betwee 0 and 1')
        fraci = 1.-frac
        # Setup distribution model and mixture model
        f, desc, n_parms, parm_names, parm_desc = self.dfit._fitlib(modal_model)
        mf = self.dfit.dist.mix_dist
        # Setup max binspace
        ibins = 1000
        BinMid = np.logspace(np.log10(range_min),np.log10(range_max),ibins)
        BinLow, BinUp = self.touching_bins(BinMid)
        dlogDp = np.log10(BinUp/BinLow)
        # Get parameterization type
        if num_parm is not None and vol_parm is None:
            parm_type = 'n'
        elif num_parm is None and vol_parm is not None:
            parm_type = 'v'
        else:
            raise Exception
        # Get dNdlogDp and dVdlogDp arrays
        if parm_type == 'n':
            dNd = mf(BinMid, f, num_parm, modes)
            # Calculate normalized volume distribution with unit conversion from nm^3 to um^3
            spam = (dNd * (np.pi/6.) * BinMid**3 * 1e-9) * dlogDp
            dVd = (spam/spam.sum()) / dlogDp
        elif parm_type == 'v':
            #dVd = np.array([mf(BinMid[b], f, vol_parm, modes) for b in range(ibins)])
            dVd = mf(BinMid, f, vol_parm, modes)
            # Calculate normalized number distribution with unit conversion from um^3 to nm^3
            spam = (dVd / (np.pi/6.) / BinMid**3 / 1e-9) * dlogDp
            dNd = (spam/spam.sum()) / dlogDp
        # Get initial lower and upper x indicies
        #max_dN = dNd.max()
        max_dN_x = dNd.argmax()
        #max_dV = dVd.max()
        max_dV_x = dVd.argmax()
        #print dNd
        #print dVd
        #print max_dN_x, max_dV_x
        bl = np.min([max_dN_x, max_dV_x])
        bu = np.max([max_dN_x, max_dV_x])
        # Expand x range and check totals
        Ntot = np.sum(dNd * dlogDp)
        Vtot = np.sum(dVd * dlogDp)
        N_ = True
        V_ = True
        while N_ or V_:
            try:
                while dNd[bl]*dlogDp[bl] >= fraci*Ntot:
                    bl-=1
                while dVd[bl]*dlogDp[bl] >= fraci*Vtot:
                    bl-=1
            except IndexError:
                bl = 0
            try:
                while dNd[bu]*dlogDp[bu] >= fraci*Ntot:
                    bu+=1
                while dVd[bu]*dlogDp[bu] >= fraci*Vtot:
                    bu+=1
            except IndexError:
                bu = ibins - 1
            # check number distribution
            if np.sum(dNd[bl:bu+1] * dlogDp[bl:bu+1])/Ntot >= frac:
                N_ = False
            # determine which direction has more of dist
            else:
                Nlow = np.sum(dNd[:bl] * dlogDp[:bl])
                Nup = np.sum(dNd[bu:] * dlogDp[bu:])
                if Nlow > Nup:
                    bl-=1
                elif Nlow < Nup:
                    bu+=1
                else:
                    bl = np.max(bl-1, 0)
                    bu = np.min(bu+1, ibins-1)
            # check volume distribution
            if np.sum(dVd[bl:bu+1] * dlogDp[bl:bu+1])/Vtot >= frac:
                V_ = False
            # determine which direction has more of dist
            else:
                Vlow = np.sum(dVd[:bl] * dlogDp[:bl])
                Vup = np.sum(dVd[bu:] * dlogDp[bu:])
                if Vlow > Vup:
                    bl-=1
                elif Vlow < Vup:
                    bu+=1
                else:
                    bl = np.max(bl-1, 0)
                    bu = np.min(bu+1, ibins-1)

        # return new size distribution
        return np.logspace(np.log10(BinMid[bl]),np.log10(BinMid[bu]),nbins)

    def _param_dist(self, num_parm=None, vol_parm=None,
                        CNconc=None, Volconc=None,
                        modes=1, modal_model='LognormPDFv',
                        data_obj=None, auto_bins=False, nbins=200):
        """Creates a parameterized size distribution and associated variables for one or more modes, for each datapoint.
        Note that this will override any size distribution already associated with each datapoint in parent, but
        will apply to this instance of CNdist only. It should typically be called at instantiation of the class (hence
        the nominal private method).
        The dNdlogDp variable within self.data will be the combined distribution for all modes, while _# will be
        appended to each individual modal variable.
        The parameters will be stored as ._p_parms, ._p_CNconc, ._p_modes, ._p_modal_model when implemented.
        The input values (parms, CNconc) can be arrays for each datapoint, or can be grabbed as arrays
        from an already imported variable. If importing, set variable to a string as self.data.__dict__[string].
        parms are assumed to be for normalized size distributions and will multiply the amplitude by CNconc parameter.
        modes is fixed, so should be the maximum number of modes for a datapoint. The amplitude/modal fraction parameter
        can be set to zero for a datapoint to leave the mode out (all dNdlogDp are zero).
        modal_model is fixed for now.
        Note that this method also runs norm_dNdlogDp for the full distribution and each mode automatically if
        norm is True (default).

        Currently, this method allows for parameterization using either number or volume distribution parameters.
        These distribution parameters are passed via the num_parm or vol_parm method parameters, and should match
        the format used by modal_model.

        Parameters:
          Only one of:
            num_parm:       A 1D list of number distribution parameters to keep the same parameters for all datapoints, or
                            a 2D list of number distribution parameters of size self.data.n for different parameters
                            for each datapoint.
            vol_parm:       A 1D list of volume distribution parameters to keep the same parameters for all datapoints, or
                            a 2D list of volume distribution parameters of size self.data.n for different parameters
                            for each datapoint.
          Only one of:
            CNconc:         The total number concentration of the population, or a 1D list of number concentrations
                            for each datapoint.
            Volconc:        The total volume concentration of the population, or a 1D list of volume concentrations
                            for each datapoint.
        Optional Parameters:
            modes:          The maximum number of modes in the distribution to be generated.
            modal_model:    The modal model, typically lognormal, to generate the distributions from.
            auto_bins:      If True, will create a new set of lognormally spaced initial bins that
                            capture most of size distribution. Low end is defined as the diameter
                            where dN/dlogDp = 0.001 * max(dN/dlogDp) and the upper end
                            where dV/dlogDp = 0.001 * max(dV/dlogDp). If BinMid, or BinLow and BinUp, are
                            not given, will set this to True.
            nbins:          The number of bins to use if auto_bins is True.
        """
        # -Setup data object-
        if data_obj is None:
            data_obj = self.data
        ndp = self.ndp

        # -Get distribution parameters- (eventually surface area is possible as well)
        # Number parameterization
        if num_parm is not None:
            if vol_parm is not None:
                raise Exception('Only one type of distribution parameterization allowed')
            parm_type = 'n'
            if isinstance(num_parm, str):
                parms = data_obj.__dict__[num_parm].d
            else:
                parms = num_parm
        # Volume parameterization
        elif vol_parm is not None:
            if num_parm is not None:
                raise Exception('Only one type of distribution parameterization allowed')
            parm_type = 'v'
            if isinstance(vol_parm, str):
                parms = data_obj.__dict__[vol_parm].d
            else:
                parms = vol_parm
        # If neither number or volume parms are passed, look to see if a parm set
        elif hasattr(self, '_p_parms'):
            parms = self._p_parms
            parm_type = self._p_parmtype
        else:
            raise Exception('No parameter list found')
        # Concentrations at each datapoint
        parms = np.array(parms)
        if parms.ndim == 1:
            # constant size dist at all datapoints
            parms = np.row_stack([parms]*ndp)
        if parms.shape[0] != ndp:
            raise Exception('Parameter variable has wrong shape: %s, %s'%(str(parms.shape), ndp))

        # -Get distribution concentration-
        set_data_CNconc = True
        set_data_Volconc = True
        # Number concentration constrained distribution
        if CNconc is not None:
            if Volconc is not None:
                raise Exception('Only one type of distribution concentration allowed')
            conc_type = 'n'
            if isinstance(CNconc, str):
                concs = data_obj.__dict__[CNconc].d
                set_data_CNconc = False
            else:
                concs = CNconc
        # Volume conentration constrained distribution
        elif Volconc is not None:
            if CNconc is not None:
                raise Exception('Only one type of distribution concentration allowed')
            conc_type = 'v'
            if isinstance(Volconc, str):
                concs = data_obj.__dict__[Volconc].d
                set_data_Volconc = False
            else:
                concs = Volconc
        # Neither given, so look for existing conc object
        else:
            if hasattr(data_obj, 'CNconc'):
                conc_type = 'n'
                concs = data_obj.CNconc.d
                set_data_CNconc = False
            elif hasattr(data_obj, 'Volconc'):
                conc_type = 'v'
                concs = data_obj.Volconc.d
                set_data_Volconc = False
            else:
                conc_type = 'n'
                concs = 1.
        # Concentrations at each datapoint
        concs = np.atleast_1d(concs)
        if concs.size == 1:
            # constant concentration at all datapoints
            concs = np.row_stack([concs]*ndp)
        if concs.shape[0] != ndp:
            raise Exception('Concentration variable has wrong shape: %s'%concs.shape)

        # -Setup Model and Variables-
        # Get Bin variables
        if auto_bins:
            BinMid = None
            dlogDp = None
        else:
            BinMid = data_obj.BinMid.d
            nbins = data_obj.BinMid.length
            dlogDp = data_obj.dlogDp.d
        # Setup distribution model and mixture model
        f, desc, n_parms, parm_names, parm_desc = self.dfit._fitlib(modal_model)
        mf = self.dfit.dist.mix_dist
        # Determine if modal analysis is needed (for multimodal distributions)
        if modes > 1:
            run_modes = True
        else:
            run_modes = False
        # Setup modal normalized size distribution arrays as dimensions: [datapoint, mode, bin]
        if run_modes:
            modal_norm_dNdlogDp = np.zeros([ndp, modes, nbins])
            modal_norm_dNdlogDp[:] = np.NaN
            modal_norm_dVdlogDp = np.zeros([ndp, modes, nbins])
            modal_norm_dVdlogDp[:] = np.NaN
        # Setup modal size distribution arrays as dimensions: [datapoint, mode, bin]
        if run_modes:
            modal_dNdlogDp = np.zeros([ndp, modes, nbins])
            modal_dNdlogDp[:] = np.NaN
            modal_dVdlogDp = np.zeros([ndp, modes, nbins])
            modal_dVdlogDp[:] = np.NaN
        # Setup total normalized size distribution arrays as dimensions: [datapoint, bin]
        norm_dNdlogDp = np.zeros([ndp, nbins])
        norm_dNdlogDp[:] = np.NaN
        norm_dVdlogDp = np.zeros([ndp, nbins])
        norm_dVdlogDp[:] = np.NaN
        # Setup total size distribution arrays as dimensions: [datapoint, bin]
        dNdlogDp = np.zeros([ndp, nbins])
        dNdlogDp[:] = np.NaN
        dVdlogDp = np.zeros([ndp, nbins])
        dVdlogDp[:] = np.NaN
        # Setup modal concentration and modal fraction arrays as dimensions: [datapoint, mode]
        if run_modes:
            modal_CNconc = np.zeros([ndp, modes])
            modal_CNconc[:] = np.NaN
            modal_Volconc = np.zeros([ndp, modes])
            modal_Volconc[:] = np.NaN
            modal_numfrac = np.zeros([ndp, modes])
            modal_numfrac[:] = np.NaN
            modal_volfrac = np.zeros([ndp, modes])
            modal_volfrac[:] = np.NaN
        # Setup concentration arrays as dimensions: [datapoint]
        CNconc = np.zeros([ndp])
        CNconc[:] = np.NaN
        Volconc = np.zeros([ndp])
        Volconc[:] = np.NaN
        # Get parameter location for amplitude (typically N, but haven't standardized this yet... #TODO:)
        ploc = np.where(np.array(parm_names) == 'N')[0][0]

        # -Generate distributions and concentrations for each data point and mode-
        # For each data point
        for n in range(ndp):
            # Reshape parms
            parms_ = np.reshape(parms[n], [modes,n_parms])
            # Ensure parms modal fractions sum to 1
            mfsum = np.sum(parms_, axis=0)[ploc]
            if mfsum != 1.:
                parms_[:,ploc] = parms_[:,ploc] / mfsum
                mfsum = np.sum(parms_, axis=0)[ploc]
            # Normalized parms for modal norm dists
            modal_parms = parms_.copy()
            modal_parms[:,ploc] = 1.

            # Generate bins if needed
            if auto_bins:
                if parm_type == 'n':
                    BinMid = self.generate_bins(num_parm=parms[n], modes=modes, modal_model=modal_model, nbins=nbins)
                elif parm_type == 'v':
                    BinMid = self.generate_bins(vol_parm=parms[n], modes=modes, modal_model=modal_model, nbins=nbins)
                BinLow, BinUp = self.touching_bins(BinMid)
                dlogDp = np.log10(BinUp/BinLow)
                self._setup_dataobj(data_obj, gvar={'BinMid':BinMid, 'BinUp':BinUp, 'BinLow':BinLow, 'dlogDp':dlogDp})

            # Create normalized distributiomf(BinMid[b], f, parms[n], modes)n variables for total population, and each mode if needed
            if parm_type == 'n':
                norm_dNdlogDp[n] = [mf(BinMid[b], f, parms[n], modes) for b in range(nbins)]
                # Calculate normalized volume distribution with unit conversion from nm^3 to um^3
                spam = (norm_dNdlogDp[n] * (np.pi/6.) * BinMid**3 * 1e-9) * dlogDp
                norm_dVdlogDp[n] = (spam/spam.sum()) / dlogDp
                if run_modes:
                    modal_norm_dNdlogDp[n] = [[f(BinMid[b], *modal_parms[m]) for b in range(nbins)] for m in range(modes)]
                    # Calculate normalized volume distribution with unit conversion from nm^3 to um^3
                    spam = [(modal_norm_dNdlogDp[n,m] * (np.pi/6.) * BinMid**3 * 1e-9) * dlogDp for m in range(modes)]
                    modal_norm_dVdlogDp[n] = [(spam[m]/spam[m].sum()) / dlogDp for m in range(modes)]
                    # Calculate number and volume modal fractions
                    modal_numfrac[n] = [parms_[m,ploc] for m in range(modes)]
                    eggs = [(spam[m] * parms_[m,ploc]).sum() for m in range(modes)]
                    modal_volfrac[n] = [eggs[m]/np.sum(eggs) for m in range(modes)]
            elif parm_type == 'v':
                norm_dVdlogDp[n] = [mf(BinMid[b], f, parms[n], modes) for b in range(nbins)]
                # Calculate normalized number distribution with unit conversion from um^3 to nm^3
                spam = (norm_dVdlogDp[n] / (np.pi/6.) / BinMid**3 / 1e-9) * dlogDp
                norm_dNdlogDp[n] = (spam/spam.sum()) / dlogDp
                if run_modes:
                    modal_norm_dVdlogDp[n] = [[f(BinMid[b], *modal_parms[m]) for b in range(nbins)] for m in range(modes)]
                    # Calculate normalized number distribution with unit conversion from um^3 to nm^3
                    spam = [(modal_norm_dVdlogDp[n,m] / (np.pi/6.) / BinMid**3 / 1e-9) * dlogDp for m in range(modes)]
                    modal_norm_dNdlogDp[n] = [(spam[m]/spam[m].sum()) / dlogDp for m in range(modes)]
                    # Calculate number and volume modal fractions
                    modal_volfrac[n] = [parms_[m,ploc] for m in range(modes)]
                    eggs = [(spam[m] * parms_[m,ploc]).sum() for m in range(modes)]
                    modal_numfrac[n] = [eggs[m]/np.sum(eggs) for m in range(modes)]

            # Generate distribution and concentration for total and each mode depending on concentration constraint type
            if conc_type == 'n':
                # Generate number concentration variables for total population
                CNconc[n] = concs[n]
                # Generate distribution variables for total population
                dNdlogDp[n] = norm_dNdlogDp[n] * CNconc[n]
                dVdlogDp[n] = dNdlogDp[n] * (np.pi/6.) * BinMid**3 * 1e-9
                # Calculate volume concentration variables for total population
                Volconc[n] = (dVdlogDp[n] * dlogDp).sum()
                # Generate concentration and distribution variables for each mode
                if run_modes:
                    modal_CNconc[n] = [CNconc[n] * modal_numfrac[n,m] for m in range(modes)]
                    modal_dNdlogDp[n] = [modal_norm_dNdlogDp[n,m] * modal_CNconc[n,m] for m in range(modes)]
                    # Calculate volume distribution with unit conversion from nm^3 to um^3
                    modal_dVdlogDp[n] = [modal_dNdlogDp[n,m] * (np.pi/6.) * BinMid**3 * 1e-9 for m in range(modes)]
                    # Calculate volume in each mode
                    modal_Volconc[n] = [(modal_dVdlogDp[n,m] * dlogDp).sum() for m in range(modes)]
            if conc_type == 'v':
                # Generate volume concentration variables for total population
                Volconc[n] = concs[n]
                # Generate distribution variables for total population
                dVdlogDp[n] = norm_dVdlogDp[n] * Volconc[n]
                dNdlogDp[n] = dVdlogDp[n] / (np.pi/6.) / BinMid**3 / 1e-9
                # Calculate number concentration variables for total population
                CNconc[n] = (dNdlogDp[n] * dlogDp).sum()
                # Generate concentration and distribution variables for each mode
                if run_modes:
                    modal_Volconc[n] = [Volconc[n] * modal_volfrac[n,m] for m in range(modes)]
                    modal_dVdlogDp[n] = [modal_norm_dVdlogDp[n,m] * modal_Volconc[n,m] for m in range(modes)]
                    # Calculate number distribution with unit conversion from um^3 to nm^3
                    modal_dNdlogDp[n] = [modal_dVdlogDp[n,m] / (np.pi/6.) / BinMid**3 / 1e-9 for m in range(modes)]
                    # Calculate number in each mode
                    modal_CNconc[n] = [(modal_dNdlogDp[n,m] * dlogDp).sum() for m in range(modes)]

        # -Create and save variables and data objects-
        # Create a modes object to hold each mode
        if run_modes:
            setattr(self, 'modes', BlankObject())
            # Create a mode and data object for each mode and create variables in it
            for m in range(modes):
                mode_name =  'm%i'%(m+1)
                setattr(self.modes, mode_name, BlankObject('Distribution Mode %i'%(m+1)))
                setattr(self.modes.__dict__[mode_name], 'data', BlankObject())
                mode_obj = self.modes.__dict__[mode_name].data
                modal_vardict = {'dNdlogDp':modal_dNdlogDp[:,m,:], 'dVdlogDp':modal_dVdlogDp[:,m,:],
                                 'Norm_dNdlogDp':modal_norm_dNdlogDp[:,m,:], 'Norm_dVdlogDp':modal_norm_dVdlogDp[:,m,:],
                                 'CNconc':modal_CNconc[:,m], 'Volconc':modal_Volconc[:,m],
                                 'mnf':modal_numfrac[:,m], 'mvf':modal_volfrac[:,m]}
                for varname,val in modal_vardict.items():
                    self._make_var(varname, data_obj=mode_obj)
                    self._set_var(varname, val, v=np.ones(val.shape, dtype=bool), data_obj=mode_obj, vec=self._var_vec(varname))
        # Create variables for the total distribution
        total_vardict = {'dNdlogDp':dNdlogDp, 'dVdlogDp':dVdlogDp,
                         'Norm_dNdlogDp':norm_dNdlogDp, 'Norm_dVdlogDp':norm_dVdlogDp}
        if set_data_CNconc:
            total_vardict.update({'CNconc':CNconc})
        if set_data_Volconc:
            total_vardict.update({'Volconc':Volconc})
        for varname,val in total_vardict.items():
            self._make_var(varname, data_obj=data_obj)
            self._set_var(varname, val, v=np.ones(val.shape, dtype=bool), data_obj=data_obj, vec=self._var_vec(varname))

        # -Rerun _setup_dataobj for self.data and for each mode after updating _defvar to include any remaining modal calcs-
        #   Note that given how data objects are set up, I think BinMid etc, needs to be passed to each data object
        self._defvar = ['dN']
        self._setup_dataobj(data_obj)
        if run_modes:
            for m in self.modes.__dict__.values():
                self._setup_dataobj(m.data, gvar={'BinMid':data_obj.BinMid, 'BinUp':data_obj.BinUp, 'BinLow':data_obj.BinLow, 'dlogDp':data_obj.dlogDp})
        # Set a record of the parameterization as ._p attributes
        #   and a .nmodes attribute with the number of modes in the parameterization
        self._p_parms = parms
        self._p_parmtype = parm_type
        self._p_CNconc = CNconc
        self._p_modes = modes
        self._p_modal_model = modal_model
        self.nmodes = modes

    def norm_dNdlogDp(self, varname='Norm_dNdlogDp', v=None, data_obj=None, altdNkey=None, norm_vals=None, add_dN=False):
        """Creates a normalized dNdlogDp variable.
        Requires self.data.dNdlogDp and self.data.CNconc.
        If norm_vals is given, it contains an array of values to normalize against. Otherwise
        will use the data_obj.CNconc.d[i] value. Alternately, can contain a string to grab an
        array from data_obj.__dict__[norm_vals].d
        add_dN: if true, will add a 'Norm_dN' variable as well. #TODO: dif varname if str?
        """
        if data_obj is None:
            data_obj = self.data
        if altdNkey is None:
            dNdlogDp = data_obj.dNdlogDp
        else:
            dNdlogDp = data_obj.__dict__[altdNkey]
        if norm_vals is None:
            if not hasattr(data_obj, 'CNconc'):
                raise Exception
            norm_vals = data_obj.CNconc.d
            nv = True
        else:
            if isinstance(norm_vals, str):
                norm_vals = data_obj.__dict__[norm_vals].d
            if not isinstance(norm_vals, np.ndarray):
                raise Exception
            if norm_vals.size != dNdlogDp.size:
                raise Exception
            nv = False
        norm_dNdlogDp = []
        for i in range(dNdlogDp.size):
            norm_spec = []
            for j in range(dNdlogDp.length):
                norm_spec.append(dNdlogDp.d[i,j]/norm_vals[i])
            norm_dNdlogDp.append(norm_spec)
        norm_dNdlogDp = np.array(norm_dNdlogDp)
        # Add the Norm_dN variable if needed
        if add_dN:
            norm_dN = norm_dNdlogDp * data_obj.dlogDp.d
        # v is the same as dNdlogDp as long as CNconc is valid as well
        if v is None:
            v = copy.copy(dNdlogDp.v)
            if nv:
                for i in range(data_obj.CNconc.size):
                    if not data_obj.CNconc.v[i]:
                        v[i] = [False]*dNdlogDp.length
        self._make_var(varname, def_varname='Norm_dNdlogDp', data_obj=data_obj)
        self._set_var(varname, norm_dNdlogDp, v=v, vec=dNdlogDp.vec, data_obj=data_obj)
        if add_dN:
            self._make_var('Norm_dN', data_obj=data_obj)
            self._set_var('Norm_dN', norm_dN, v=v, vec=dNdlogDp.vec, data_obj=data_obj)

    def norm_dVdlogDp(self, varname='Norm_dVdlogDp', v=None, data_obj=None, altdVkey=None, norm_vals=None, add_dV=False):
        """Creates a normalized dVdlogDp variable.
        Requires self.data.dVdlogDp and self.data.CNconc.
        If norm_vals is given, it contains an array of values to normalize against. Otherwise
        will use the data_obj.Volconc.d[i] value. Alternately, can contain a string to grab an
        array from data_obj.__dict__[norm_vals].d
        add_dV: if true, will add a 'Norm_dV' variable as well. #TODO: dif varname if str?
        """
        if data_obj is None:
            data_obj = self.data
        if altdVkey is None:
            dVdlogDp = data_obj.dVdlogDp
        else:
            dVdlogDp = data_obj.__dict__[altdVkey]
        if norm_vals is None:
            if not hasattr(data_obj, 'Volconc'):
                raise Exception
            norm_vals = data_obj.Volconc.d
            nv = True
        else:
            if isinstance(norm_vals, str):
                norm_vals = data_obj.__dict__[norm_vals].d
            if not isinstance(norm_vals, np.ndarray):
                raise Exception
            if norm_vals.size != dVdlogDp.size:
                raise Exception
            nv = False
        norm_dVdlogDp = []
        for i in range(dVdlogDp.size):
            norm_spec = []
            for j in range(dVdlogDp.length):
                norm_spec.append(dVdlogDp.d[i,j]/norm_vals[i])
            norm_dVdlogDp.append(norm_spec)
        norm_dVdlogDp = np.array(norm_dVdlogDp)
        # Add the Norm_dV variable if needed
        if add_dV:
            norm_dV = norm_dVdlogDp * data_obj.dlogDp.d
        # v is the same as dNdlogDp as long as CNconc is valid as well
        if v is None:
            v = copy.copy(dVdlogDp.v)
            if nv:
                for i in range(data_obj.Volconc.size):
                    if not data_obj.Volconc.v[i]:
                        v[i] = [False]*dVdlogDp.length
        self._make_var(varname, def_varname='Norm_dVdlogDp', data_obj=data_obj)
        self._set_var(varname, norm_dVdlogDp, v=v, vec=dVdlogDp.vec, data_obj=data_obj)
        if add_dV:
            self._make_var('Norm_dV', data_obj=data_obj)
            self._set_var('Norm_dV', norm_dV, v=v, vec=dVdlogDp.vec, data_obj=data_obj)

    def bin_volume(self, fr=False, data_obj=None):
        """Returns Bin sizes as volumes in units of um^3.
        If fr is True, will return tuple of (BinMid, BinLow, BinUp) in volumes.
        """
        if data_obj is None:
            data_obj = self.data
        # include coversion from nm^3 to um^3
        vbm = data_obj.BinMid.d**3. * (np.pi/6.) * 1e-9
        if fr:
            vbl = data_obj.BinLow.d**3. * (np.pi/6.) * 1e-9
            vbu = data_obj.BinUp.d**3. * (np.pi/6.) * 1e-9
            return (vbm, vbl, vbu)
        return vbm

    def bin_mass(self, dens, fr=False, data_obj=None):
        """Returns Bin sizes as masses using 'dens' density in units of g.
        If fr is True, will return tuple of (BinMid, BinLow, BinUp) in masses.
        Density should be in units of g/cm^3.
        """
        if data_obj is None:
            data_obj = self.data
        # mass = volume (um^3/cm^3) * dens (g/cm^3) * (1 cm^3 / 1e12 um^3)
        if fr:
            vbm, vbl, vbu = self.bin_volume(fr=fr, data_obj=data_obj)
            mbm = vbm * dens * 1e-12
            mbl = vbl * dens * 1e-12
            mbu = vbu * dens * 1e-12
            return (mbm, mbl, mbu)
        else:
            mbm = self.bin_volume(fr=fr, data_obj=data_obj) * dens * 1e-12
            return mbm

    def touching_bins(self, binmid, low_first=True):
        """Returns a log distributed touching bin array from a numpy array of bin midpoints as 'binlow, binup'
        Arguments:
            binmid:     A numpy array of the midpoint diameters of each bin
            low_first:  If True (default), binmid is being passed from smallest to largest.
                        If False, sizes are passed from largest to smallest.
        """
        # Sizes must be given from largest to smallest for this code, so reverse given arrays if needed
        if low_first:
            if binmid[-1] <= binmid[0]:
                raise Exception(binmid)
            binmid = binmid[::-1]
        else:
            if binmid[0] <= binmid[-1]:
                raise Exception
        nbins = binmid.size
        binlow = np.zeros(nbins)
        binlow[:] = np.NaN
        binup = np.zeros(nbins)
        binup[:] = np.NaN
        # Calculate lower bin edges
        for i in range(nbins-1):
            binlow[i] = np.exp( ((np.log(binmid[i])-np.log(binmid[i+1]))/2) + np.log(binmid[i+1]) )
        i = nbins-1
        binlow[i] = binlow[i-1] * (binlow[i-1]/binlow[i-2])
        # Calculate upper bin edges
        for i in range(1,nbins):
            binup[i] = np.exp( ((np.log(binmid[i-1])-np.log(binmid[i]))/2) + np.log(binmid[i]) )
        i = 0
        binup[i] = binup[i+1] * (binup[i+1]/binup[i+2])
        if low_first:
            binlow = binlow[::-1]
            binup = binup[::-1]
        return binlow, binup


class KappaCalc(BaseAnalysis):
    """A Kappa-Kohler calculation class.
    These methods are adapted from IDL code by Markus Petters.
    Note: CCN fitting methods not included in this version.
    """
    def __init__(self, parent, var_dir, gvar={}, var={}, calc=None, ovr={}):
        """On instantiation, gets kappa lookup table, and imports needed variables from parent.
        Arguments:
            parent:     The parent of the KappaCalc() instance.
            var_dir:    The var directory that contains the pickled kappa lookup table.
            gvar:       Dictionary of global/meta variables being passed into the analysis class instance.
                        The value of each key can be a string to grab from parent.data.key,
                        or a numpy array to hold the actual values themselves.
            var:        Standard variables to be imporeted.
                        Same as gvar, but conducted after the variables in gvar.
            calc:       A dictionary of any requested calculation methods to run on instantiation.
            ovr:        An override attribute.
        """
        # Name and Description
        name = 'kappa'
        desc = ''
        # Needed analysis classes and external methods
        self.dfit = DistFit()
        self._bisect = bisect.bisect
        self._var_dir = var_dir
        self._ovr=ovr
        # Call BaseAnalisis Class init method to finish import
        BaseAnalysis.__init__(self, name, desc, parent)
        # Create default data attribute and setup variables
        setattr(self, 'data', BlankObject())
        self._setup_dataobj(self.data, gvar, var)
        # Import Dry Diameter, Critical Supersaturation, and kappa tables
        kl_file = os.path.join(self._var_dir,'kappa_lookup.pickle')
        if os.path.isfile(kl_file):
            with file(kl_file, 'rb') as f:
                import cPickle as pickle
                self._Dd, self._sc, self._k = pickle.load(f)
            self._Ddmin,self._Ddmax = self._Dd.min(), self._Dd.max()
            self._scmin,self._scmax = self._sc.min(), self._sc.max()
            self._ht = True
        else:
            self._ht = False
        # Import similar growth factor lookup tables
        gfl_file = os.path.join(self._var_dir,'gf_lookup.pickle')
        if os.path.isfile(gfl_file):
            with file(gfl_file, 'rb') as f:
                import cPickle as pickle
                self._gfRH, self._gfDd, self._gfk, self._gf = pickle.load(f)
            self._gfRHmin,self._gfRHmax = self._gfRH.min(), self._gfRH.max()
            self._gfDdmin,self._gfDdmax = self._gfDd.min(), self._gfDd.max()
            self._gfkmin,self._gfkmax = self._gfk.min(), self._gfk.max()
            self._gfht = True
        else:
            self._gfht = False
        # Run any additional requested calculations - None relevant here
        if calc is not None:
            pass

    def _var_dict(self, varname, dsname='', globveckey=False):
        """Returns VarObj info on all variables in this AnalysisClass.
        In 2D vector variable, the length of each datapoint is unknown and initially set to None.
        (typ, meas, size, length, dim, vec, units, label_axis, label_title, desc, kwargs)
        dsname: will add this to label_title and desc entries.
        This method also can return the global vector key to grab equivalent vectors from instances of this
        CNdist class in other dataset objects. If globveckey is True, will instead return the key to do this.
        """
        if varname == '':
            if globveckey:
                return None
            return (bool,False,None,1,1,None,'', '', '', '')
        else:
            # If not local, look in Base Analysis
            return super(KappaCalc, self)._var_dict(varname, dsname=dsname, globveckey=globveckey)

    def _sat_ratio(self, D, Dd, k, neg=False, sig=0.072, Mw=0.018015, R=8.315, T=298.15, rho=997.1):
        """tmp
        If neg is true, will return the negative of the saturation ratio (for minimization purposes).
        """
        if neg:
            sign = -1.0
        else:
            sign = 1.0
        return sign * ((D**3-Dd**3)/(D**3-Dd**3*(1-k))) * np.exp((4.0*sig*Mw)/(R*T*rho*D))

    def _crit_sat_ratio(self, k, Dd, sig=0.072, Mw=0.018015, R=8.315, T=298.15, rho=997.1):
        """tmp
        Uses Dd as initial guess for droplet diameter D.
        """
        rslt = sp.optimize.minimize(self._sat_ratio, Dd, args=(Dd, k, True, sig, Mw, R, T, rho), tol=1e-8, method='Nelder-Mead', options={'disp':False})
        return rslt.fun * -1.0

    def _kappa_error(self, k, Dd, ss, sig=0.072, Mw=0.018015, R=8.315, T=298.15, rho=997.1):
        rslt = np.abs(ss - self._crit_sat_ratio(k, Dd, sig=sig, Mw=Mw, R=R, T=T, rho=rho))
        return rslt

    def _Dd_error(self, Dd, k, ss, sig=0.072, Mw=0.018015, R=8.315, T=298.15, rho=997.1):
        rslt = np.abs(ss - self._crit_sat_ratio(k, Dd, sig=sig, Mw=Mw, R=R, T=T, rho=rho))
        return rslt

    def _calc(self, Dd, ss, sig=0.072, Mw=0.018015, R=8.315, T=298.15, rho=997.1, ind=None):
        """Returns iteratively calculated value of Kappa based on given values.
        Uses the numpy fmin (Nelder-Mead) algorithm to find Kappa.
        Arguments:
            Dd:     Particle dry diameter (nm)
            ss:     Supersaturation (%)
            sig:    Surface tension of solution/air interface (J/m^2)
            Mw:     Molecular weight of water (kg/mol)
            R:      Ideal gas constant (J/(K*mol))
            T:      Temperature (K)
            rho:    Density of water (kg/mol)
        """
        # Convert from nm and SS%
        Dd = Dd * 1e-9
        s = 1.0 + (ss/100.)
        D = Dd                  # Initial droplet diameter guess
        # If the function returned is not close to 0, then it didn't solve properly, try another method
        rslt = sp.optimize.fmin(self._kappa_error, 0.3, args=(D,s), disp=False)
        if rslt < 0.001:
            return 0.0
        return rslt[0]

    def _khash(self, d, ss):
        """Returns a bilinear interpolated kappa value from the kappa lookup table.
        If d or ss is outside of their respective lookup table ranges, a NaN is returned.
        Bilinear interpolation uses only the four kappa values around the requested
        data point (The two values it is between in each direction of d and ss).
        d in nm.
        """
        if self._ht is not True:
            return np.NaN
        if d < self._Ddmin or d >= self._Ddmax or ss < self._scmin or ss >= self._scmax:
            return np.NaN
        xi = self._bisect(self._Dd,d)
        yi = self._bisect(self._sc,ss)
        x1, x2 = self._Dd[xi-1], self._Dd[xi]
        y1, y2 = self._sc[yi-1], self._sc[yi]
        return ( (1./((x2-x1)*(y2-y1))) * (
                (self._k[xi-1,yi-1]*(x2-d)*(y2-ss)) +
                (self._k[xi,yi-1]*(d-x1)*(y2-ss)) +
                (self._k[xi-1,yi]*(x2-d)*(ss-y1)) +
                (self._k[xi,yi]*(d-x1)*(ss-y1))
               ) )

    def k(self, d, ss, khash=True, nocalc=False, ind=None, verbose=False, maxk=None):
        """Returns the kappa value for a given diameter and supersaturation.
        Arguments:
            d:          Particle dry diameter (nm)
            ss:         Critical supersaturation
            khask:      If True, will first try to use the kappa lookup table at self._khash
                        If False, will interatively calculate kappa with self._calc
            nocalc:     If True, will return NaN when retrieval outside lookup table
            ind:        Index of run for warning output and debugging information only
            verbose:    If True, will print warning when kappa is outside lookup table
            maxk:       If set to a value, will return NaN above this value
        """
        if np.isnan(d).any() or np.isnan(ss).any():
            return np.NaN
        if isinstance(d, np.ndarray) or isinstance(d, list):
            d = d[0]
        if isinstance(ss, np.ndarray) or isinstance(ss, list):
            ss = ss[0]
        if self._ht and khash:
            r = self._khash(d,ss)
            if np.isnan(r):
                if nocalc:
                    r = np.NaN
                else:
                    r = self._calc(d,ss)
                if verbose:
                    print 'Kappa retrieval outside lookup table'
                    if ind is None:
                        print 'ss:', ss, 'd:', d, 'calc:', r
                    else:
                        print 'ss:', ss, 'd:', d, 'calc:', r, 'ind:', ind
        else:
            r = self._calc(d,ss)
        if maxk is not None:
            if r > maxk:
                return np.NaN
        return r

    def crit_Dd(self, k, ss, sig=0.072, Mw=0.018015, R=8.315, T=298.15, rho=997.1):
        """Returns the critical diameter (nm) for a given kappa value and supersaturation.
        Uses the numpy fmin (Nelder-Mead) algorithm to find crit Dd.
        Arguments:
            k:      Particle Kappa (-)
            ss:     Supersaturation (%)
            sig:    Surface tension of solution/air interface (J/m^2)
            Mw:     Molecular weight of water (kg/mol)
            R:      Ideal gas constant (J/(K*mol))
            T:      Temperature (K)
            rho:    Density of water (kg/mol)
        """
        # Convert from SS%
        s = 1.0 + (ss/100.)
        # Minimize dry diameter error for a given supersaturation using initial guess of 50 nm (in units of m)
        rslt = sp.optimize.fmin(self._Dd_error, 50e-9, args=(k,s), disp=False, ftol=1e-9)
        return rslt[0]*1e9

    def crit_ss(self, Dd, k, sig=0.072, Mw=0.018015, R=8.315, T=298.15, rho=997.1):
        """Returns the critical supersaturation (%) for a given kappa value and dry diameter.
        Uses the numpy fmin (Nelder-Mead) algorithm to find crit ss.
        Arguments:
            k:      Particle Kappa (-)
            Dd:     Particle Dry Diameter (nm)
            sig:    Surface tension of solution/air interface (J/m^2)
            Mw:     Molecular weight of water (kg/mol)
            R:      Ideal gas constant (J/(K*mol))
            T:      Temperature (K)
            rho:    Density of water (kg/mol)
        """
        # Convert from nm to m
        Dd = Dd * 1e-9
        # Minimize dry diameter error for a given supersaturation using initial guess of 50 nm (in units of m)
        rslt = self._crit_sat_ratio(k, Dd, sig=sig, Mw=Mw, R=R, T=T, rho=rho)
        return (rslt-1.)*100.

    def _gfhash(self, RH, Dd, k):
        """Returns a bilinear interpolated growth factor value from the gf lookup table.
        If variables are outside of their respective lookup table ranges, a NaN is returned.
        Bilinear interpolation uses only the values around the requested
        data point (The two values it is between in each direction of each variable).
        Dd in nm.
        """
        if self._gfht is not True:
            return np.NaN
        if RH < self._gfRHmin or RH >= self._gfRHmax or \
           Dd < self._gfDdmin or Dd >= self._gfDdmax or \
           k < self._gfkmin or k >= self._gfkmax:
            return np.NaN
        xi = self._bisect(self._gfRH,RH)
        yi = self._bisect(self._gfDd,Dd)
        zi = self._bisect(self._gfk,k)
        x1, x2 = self._gfRH[xi-1], self._gfRH[xi]
        y1, y2 = self._gfDd[yi-1], self._gfDd[yi]
        z1, z2 = self._gfk[zi-1], self._gfk[zi]
        return ( (1./((x2-x1)*(y2-y1)*(z2-z1))) * (
                (self._gf[xi-1,yi-1,zi-1]*(x2-RH)*(y2-Dd)*(z2-k)) +
                (self._gf[xi,yi-1,zi-1]*(RH-x1)*(y2-Dd)*(z2-k)) +
                (self._gf[xi-1,yi,zi-1]*(x2-RH)*(Dd-y1)*(z2-k)) +
                (self._gf[xi,yi,zi-1]*(RH-x1)*(Dd-y1)*(z2-k)) +
                (self._gf[xi-1,yi-1,zi]*(x2-RH)*(y2-Dd)*(k-z1)) +
                (self._gf[xi,yi-1,zi]*(RH-x1)*(y2-Dd)*(k-z1)) +
                (self._gf[xi-1,yi,zi]*(x2-RH)*(Dd-y1)*(k-z1)) +
                (self._gf[xi,yi,zi]*(RH-x1)*(Dd-y1)*(k-z1))
               ) )

    def gf(self, RH, Dd, k, sig=0.072, Mw=0.018015, R=8.315, T=298.15, rho=997.1):
        """Returns the kappa-Kohler theory estimate (PK07 eqn 6) for subsaturated growth factor.
        Requires the RH, particle dry diameter, and particle kappa value.
        Dd in nm.
        """
        if np.isnan([RH, Dd, k]).any():
            return np.NaN
        if k <= 0.:
            return np.NaN
        if RH > 100.:
            return np.NaN
        # Try lookup table
        if self._gfht:
            gf = self._gfhash(RH, Dd, k)
            if not np.isnan(gf):
                return gf
        # if lookup table failed, calc
        # Convert RH to saturation ratio
        rh = RH/100.
        # convert from nm to m
        Dd = Dd * 1.0e-9
        # Initial guess for droplet diameter is Dd*1.1
        sr = lambda D,Dd,k,rh: np.abs(rh - self._sat_ratio(D,Dd,k, sig=sig, Mw=Mw, R=R, T=T, rho=rho))
        rslt = sp.optimize.fmin( sr, Dd*1.1, args=(Dd,k,rh), disp=False )
        gf = rslt[0]/Dd
        return gf

    def _create_kappa_lookup_table(self, save_dir=None):
        """Creates a kappa lookup table.
        Note: Took about 13 hours to run on a standard desktop computer.
        """
        if save_dir is not None:
            sd = save_dir
        else:
            sd = self._var_dir
        Dd = np.logspace(1,3,1000)          # Particle dry diameter
        sc = np.array(range(1,201))*0.01    # Particle critical supersaturation
        k = np.zeros([Dd.size,sc.size])     # kappa array
        for i in range(Dd.size):
            for j in range(sc.size):
                k[i,j] = self._calc(Dd[i],sc[j])

        with file(os.path.join(sd,'kappa_lookup.pickle'), 'wb') as f:
            pickle.dump((Dd,sc,k),f,pickle.HIGHEST_PROTOCOL)

    def _create_gf_lookup_table(self, save_dir=None):
        """Creates a growth factor lookup table.
        RH 0% to 99.99%
        Dd 5 nm to 10 um
        k 0.01 to 1.00 in 0.01 increments
        """
        if save_dir is not None:
            sd = save_dir
        else:
            sd = self._var_dir
        RH = 100.*(1. - np.logspace(-4,0,200))[::-1]        # Env RH
        Dd = np.logspace(np.log10(5),np.log10(10000),1000)  # Particle dry diameter
        k = np.arange(0.01,1.01,0.01)                       # kappa array
        gf = np.array([[[self.gf(RH_, Dd_, k_) for k_ in k] for Dd_ in Dd] for RH_ in RH])

        with file(os.path.join(sd,'gf_lookup.pickle'), 'wb') as f:
            pickle.dump((RH,Dd,k,gf),f,pickle.HIGHEST_PROTOCOL)


class Optical(BaseAnalysis):
    """A class to calculate optical properties of aerosol populations.

    Will primarily interact with CNdist and KappaCalc classes to humidify populations
    that are generally given by size distribution and hygroscopicity properties.
    Scattering, absorption, and extinction properties are calculated using Mie code,
    and total path extinction and aerosol optical depths can be calculated by integration
    over given path lengths.

    For analysis of aerosol population optical properties:
    As the optical class operates based on using fundamental size, composition, and relative humidity
    information for one or more aerosol populations, each datapoint (either by time, or by some other
    parameter such as height or location) requires at least one of the dN/dlogDp and kappa variables,
    or a parameterization of these.
    When analyzing a population for optical characteristics using this class, the size and composition
    of one or more modes will be tracked within an instance of this class. It then will analyze and
    grow it using CNdist and KappaCalc at the parent (assumes they already exist for now).

    Additional calculations of AOD can be conducted by including path variables which specific a set
    of datapoints than constitute a path (e.g. a vertical column) through which a AOD can be calculated.

    NOTE: This version of the Optical() class is simplified with the intention of being used primarily
    for parameterized aerosol populations. As a result, most validity checking of data points is off,
    and only limited checks of the appropriateness for some of these calculations are included.

    """
    def __init__(self, parent, gvar={}, var={}, calc=None, name_ovr=None, obj_setpt=None,
                 CNdist=None, RH=None, data_obj_name='data',
                 mie_cache_size=100000):
        """
        By default, will clear the mie cache at the end of initial calculations.
        Parameters:
            parent:     The parent of the Optical() instance.
            gvar:       Dictionary of global/meta variables being passed into the analysis class instance.
                        The value of each key can be a string to grab from parent.data.key,
                        or a numpy array to hold the actual values themselves.
            var:        Standard variables to be imported.
                        Same as gvar, but conducted after the variables in gvar.
            calc:       An (optionally ordered) dictionary of any requested calculation methods to run on instantiation.
            name_ovr:   Will override the name of the analysis class instance. Should typically be used only when
                        creating a second or non-typical version of this type of distribution analysis.
            obj_setpt:  Allows for the instance to be set at a different point from the dataset parent.
                        Typically, for adding an interim object to store multiple instances together in
                        a unique namespace (e.g. All Optical() instances found in self.optical.__dict__).
            mie_cache_size:     The size of the cache to use in the pymiecoated instance.
        Properties:
            These are a new way of assigning data to an analysis class instance that I'm trying and are
            somewhat experimental (from a code perspective). They are stored in the self.props attribute
            and function as references to objects needed by the class instance.

            As the CNdist, kappa, dry_m, and CNconc properties are all linked, each of these variables should be
            found within the CNdist object passed to this class instance. This should simplify things by completely
            segregating the aerosol population variables from the optical class. It also means the CNdist must be
            fully setup before instantiating this class, and things like CNconc or kappa can't be changed within
            this class. In addtion, as the CNdist is a single object, it should make it easier to include multiple
            aerosol populations, each as their own CNdist instance. Multiple CNdist instances will result in their
            impacts being combined in the overall calculations.
            Each CNdist will be stored as CNdist* wildcard attributes within self.props.

            CNdist:     A list of CNdist instances that each include the following data variables:
                ndp:        The number of datapoints in the CNdist (must be the same for all CNdist).
                nmodes:     The number of modes in the distribution.
                kappa:      A 2D variable of shape [# dp, # modes] with the kappa value of each mode.
                dry_m:      A 2D variable of shape [# dp, # modes] with the complex index of refraction
                            for dry particles for each mode.
                CNconc:     A 1D variabe of shape [# dp] of total population CN number concentration
                            for this CN dist population type at each datapoint.
            RH:         A variable object of 1D (# dp) of environmental relative humidity values for each
                        datapoint. This value is applied to each CNdist population for growth calculations.
                        If values are to be passed, set as a variable in var for import and pass the name
                        of the variable. If RH is left as None, will assume the instance has a variable
                        named 'RH' where it will look for this property.
            ndp:        The number of datapoints (the same for all CNdist instances).
            npop:       The number of aerosol populations (CNdist instance) for this optical reconstruction.
            data_obj_name: The name of the data object in self, and in CNdist. Default is 'data', but could be a
                        reduced dataset such as 'red'. Note that only one data object should exist within an
                        instance of this class.
                        #XXX: I haven't done much testing on how this actually would work and how it is implemented,
                        so it may not all quite work yet for data objects other than 'data'.
                        Right now I'm thinking I'll create a shadow data object self._data. Whatever name is in
                        data_obj_name will point to self._data, e.g. self.red = self._data.

            #TODO: update how CNdist propagates kappa and dry_m variables through to each mode. At least ensure
            it is the same as was being done by this class.

        Saved variables for multiple aerosol populations:
            In the case of multiple aerosol populations being passed (multiple CNdist instances), will create
            a self.pops object that holds self.pops.p#._data objects for each population. The self._data data
            object will contain these variables for the total of all aerosol populations.
        """
        # Name and Description
        name = 'opt'
        desc = 'Aerosol Optical Analysis'
        if name_ovr is not None:
            name = name_ovr
        # Create pymiecoated Mie instances to use throughout this instance
        #   Note: self.mie for no coated calcs, self.miecoat for coated calcs
        self._mie_cache_size = mie_cache_size
        self.mie = pym.Mie(cache_size=self._mie_cache_size)
        self.miecoat = pym.Mie(cache_size=self._mie_cache_size)
        # Needed analysis classes
        # Call BaseAnalysis Class init method to finish import
        BaseAnalysis.__init__(self, name, desc, parent, obj_setpt=obj_setpt)
        # List of default calculated variables all instances of this class will have
        self._defgvar = []
        self._defvar = []
        # Scrub certain types of input variables by proper dimension
        if 'RH' in var:
            var['RH'] = np.atleast_1d(var['RH'])
        # Create default data attribute and setup variables
        setattr(self, '_data', BlankObject())
        self._setup_dataobj(self._data, gvar, var)
        self.data_obj_name = data_obj_name
        # Set properties and self.pops data objects if more than one aerosol population
        self._set_props(CNdist, RH)
        # Run any additional requested calculations
        if calc is not None:
            for calc_name, calc_args in calc.items():
                # To allow for multiple calls to the same method, using fnmatch
                if fnmatch.fnmatch(calc_name, 'ext_recon*'):
                    # Extinction coefficient at each datapoint
                    self.ext_recon(**calc_args)
                elif fnmatch.fnmatch(calc_name, 'path_AOD*'):
                    # Calculate AOD along a given path
                    self.path_AOD(**calc_args)
                else:
                    raise Exception('Calculation name not found')
            self._clear_mie_cache()

    def _var_dict(self, varname, dsname='', globveckey=False):
        """Returns VarObj info on all variables in this AnalysisClass.
        In 2D vector variable, the length of each datapoint is unknown and initially set to None.
        (typ, meas, size, length, dim, vec, units, label_axis, label_title, desc, kwargs)
        dsname: will add this to label_title and desc entries.
        This method also can return the global vector key to grab equivalent vectors from instances of this
        class in other dataset objects. If globveckey is True, will instead return the key to do this.
        """
        if varname == '':
            if globveckey:
                return None
            return (bool,False,None,1,1,None,'', '', '', '')
        elif varname == 'loc':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(m)$','Grid Box Location',dsname+'Grid Box Location',dsname+'Grid Box Location')
        elif varname == 'RH':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(\%)$','Model RH',dsname+'Model RH',dsname+'Model RH')
        elif varname == 'CNparms':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(-)$','',dsname+'',dsname+'')
        elif varname == 'gf':
            if globveckey:
                return None
            return (float,True,None,None,2,None,'$(-)$','gf',dsname+'Growth Factor',dsname+'Growth Factor')
        elif varname == 'bext':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(Mm^{-1})$','Extinction Coef',dsname+'Extinction Coefficient',dsname+'Extinction Coefficient')
        elif varname == 'dbextdlogDp':
            if globveckey:
                return None
            return (float,True,None,None,2,None,'$(Mm^{-1} cm^{-3})$','$db_{ext}/dlogD_p$',dsname+'Extinction Coefficient Distribution',dsname+'Extinction Coefficient Distribution')
        elif varname == 'Norm_dbextdlogDp':
            if globveckey:
                return 'BinMid'
            return (float,True,None,None,2,None,'$(cm^{-3})$','$d\widetilde{b_{ext}}/dlogD_p$',dsname+'Normalized dbext/dlogDp',dsname+'Normalized dbext/dlogDp')
        elif varname == 'AOD':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(-)$','AOD',dsname+'Path AOD',dsname+'Total Path AOD')
        elif varname == 'AOD_grid':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(-)$','AOD',dsname+'Grid Box AOD',dsname+'Grid Box AOD')
        elif varname == 'AOD_cumgrid':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(-)$','Cumulative AOD',dsname+'Grid Box Cumulative AOD',dsname+'Grid Box Cumulative AOD')
        elif varname == 'AOD_frac':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(-)$','AOD Fraction',dsname+'Grid Box AOD Fraction',dsname+'Grid Box AOD Fraction')
        elif varname == 'AOD_cumfrac':
            if globveckey:
                return None
            return (float,True,None,1,1,None,'$(-)$','AOD Cumulative Fraction',dsname+'Grid Box AOD Cumulative Fraction',dsname+'Grid Box AOD Cumulative Fraction')
        else:
            # If not local, look in Base Analysis
            return super(Optical, self)._var_dict(varname, dsname=dsname, globveckey=globveckey)

    def _set_props(self, CNdist_, RH):
        """Sets the CNdist property and associated attributes based on CNdist input.
        """
        # Create property object attribute and set referenced property objects
        setattr(self, 'props', BlankObject())
        # set list of CNdist aerosol populations
        setattr(self.props, 'CNdist', BlankObject())
        popn = 1
        if isinstance(CNdist_, list) or isinstance(CNdist_, np.ndarray):
            for cnd in CNdist_:
                if not isinstance(cnd, CNdist):
                    raise Exception('Entry in CNdist argument not an instance of CNdist class')
                if cnd.__name__ != 'cn':
                    pop_name = cnd.__name__
                else:
                    pop_name = 'p%i'%(popn+1)
                setattr(self.props.CNdist, pop_name, cnd)
                popn+=1
        elif isinstance(CNdist_, CNdist):
            setattr(self.props.CNdist, CNdist_.__name__, CNdist_)
        else:
            raise Exception('CNdist argument not an instance of CNdist class')
        # size and number of CNdist
        self.props.npop = len(self.props.CNdist.__dict__.keys())
        self.props.ndp = self.props.CNdist.__dict__.values()[0].ndp
        for cnd in self.props.CNdist.__dict__.values():
            if cnd.ndp != self.props.ndp:
                raise Exception('Incorrect ndp size in a CNdist instance')
        # set RH
        if isinstance(RH, VarObject):
            self.props.RH = RH
        elif RH is None:
            self.props.RH = self._data.RH
        elif isinstance(RH, str):
            self.props.RH = self._data.__dict__[RH]
        else:
            raise Exception('RH variable incorrect value')

        # If more than one aerosol population, setup population objects in self.pops
        if self.props.npop > 1:
            setattr(self, 'pops', BlankObject())
            for cnd in self.props.CNdist.__dict__.values():
                setattr(self.pops, cnd.__name__, BlankObject())
                setattr(self.pops.__dict__[cnd.__name__], '_data', BlankObject())
                # Set data_obj_name to point to pop_obj._data.
                self.pops.__dict__[cnd.__name__].__dict__[self.data_obj_name] = self.pops.__dict__[cnd.__name__]._data

        # Set data_obj_name to point to self._data.
        self.__dict__[self.data_obj_name] = self._data

    def _ind_mix(self, Dd, D, md, mm=np.complex(1.33, 0.0)):
        """Returns the adjusted index of refraction based on the volume mixing rule.
        Default is for mixing with water (e.g. hygroscopic growth). Assumes spherical particles.
        Parameters:
            Dd:         Dry diameter of the first component of the particle.
                        Alternatively, the diameter of the pure particle of component 1.
            D:          Humidified droplet diameter.
                        Alternatively, the diameter of the mixture of component 1 and 2.
            md:         Complex index of refraction of the dry particle
                        Alternatively, the complex index of refraction of component 1.
            mm:         Complex index of refraction of water (default at 550 nm)
                        Alternatively, the complex index of refraction of component 2.
        """
        V = (1./6.) * np.pi * D**3
        Vd = (1./6.) * np.pi * Dd**3
        Vm = V - Vd
        n = (Vd/V) * md + (Vm/V) * mm
        return n

    def _clear_mie_cache(self):
        """Clears the mie cache to free up memory.
        """
        delattr(self.mie, '_cache')
        delattr(self.miecoat, '_cache')
        setattr(self.mie, '_cache', pym.mie_aux.Cache(self._mie_cache_size))
        setattr(self.miecoat, '_cache', pym.mie_aux.Cache(self._mie_cache_size))
        gc.collect()

    def _S12_calc(self, wl, Dp, costh, m=np.complex(1.53, 0.0), DpC=None, mC=None):
        """Calculates particle Mie theory amplitude scattering matrix components S1 and S2.
        Assumes spherical particles and cores/shells.
        NOTE: the imaginary component of the index of refraction (absorption) should be
        positive, e.g. opposite of actual value.
        Arguments:
            Dp:     Diameter of particles (nm) or shell diameter
            wl:     Wavelength of light for scattering calc (nm)
            costh:  Cosine of the scattering angle (range: -1 <= costh <=1)
            m:      Complex refractive index or shell
            DpC:    Core Diameter of particles (nm) or None
            mC:     Core complex refractive index or None
        Returns:
            tuple of (S1, S2)
        """
        if np.isnan([wl, Dp]).any():
            return np.NaN
        # Single index of refraction
        if DpC is None:
            # Get Shape Parameter and Scattering Efficiency
            alpha = np.pi * Dp / wl
            self.mie._set_x(alpha)
            self.mie._set_m(m)
            return self.mie.S12(costh)
        else:
            # Get Shape Parameter and Scattering Efficiency
            alpha1 = np.pi * Dp / wl
            alpha2 = np.pi * DpC / wl
            self.miecoat._set_y(alpha1)
            self.miecoat._set_x(alpha2)
            self.miecoat._set_m(mC)
            self.miecoat._set_m2(m)
            return self.miecoat.S12(costh)

    def _scat_intens_func(self, theta, wl, Csca):
        """Scattered intensity calculated from amplitude function.
        theta in radians
        Assumes spherical particles, etc...
        NOTE: Does not set mie parameters! Size and complex index of
        refraction parameters !!MUST!! be set before calling this function.
        """
        S1, S2 = self.mie.S12(np.cos(theta))
        return ((wl**2/(2*np.pi*Csca)) * (np.abs(S1)**2 + np.abs(S2)**2))

    def scat_phase_func(self, theta, wl, Dp, m):
        """Calculates the single particle scattering phase function.
        Assumes spherical particles.
        Units of wavelength and particle diameter should be the same.
        NOTE: core shell version not setup yet.
        Note that units for both wl and Dp should be the same
        Parameters:
            theta:  Angle of the phase function relative to incident light (deg)
            wl:     Wavelength of light for scattering calc
            Dp:     Diameter of particles or core diameter
            m:      Complex refractive index or core
        """
        # Set Size Parameter and ref ind in self.mie
        alpha = np.pi * Dp / wl
        self.mie._set_x(alpha)
        self.mie._set_m(m)
        qsca = self.mie.qsca()
        Csca = qsca * np.pi * (Dp/2.)**2
        # Calculate intensity at desired angle
        if np.isscalar(theta):
            P = self._scat_intens_func(np.deg2rad(theta),wl,Csca)
        else:
            P = np.array([self._scat_intens_func(ti,wl,Csca) for ti in np.deg2rad(theta)])

        return P

    def _leg_poly_calc(self, x, l):
        """Zero indexed function to return value of lth term of l-order legendre polynomial at x."""
        # TODO: Very inefficient way of doing this. Need to find a better (preferably built in) solution
        return np.polynomial.legendre.legval(x,[0.]*int(l)+[1])

    def _leg_coeff_func(self, costh, l, wl, Dp, m):
        return self.scat_phase_func(np.rad2deg(np.arccos(costh)), wl, Dp, m) * self._leg_poly_calc(costh, l)

    def _leg_coeff_calc(self, l, wl, Dp, m):
        return 1./2. * sp.integrate.quad(self._leg_coeff_func, -1., 1., args=(l,wl,Dp,m))[0]

    def legendre_coeffs(self, wl, Dp, m, nc=None, alt=True):
        """Computes the Legendre Coefficients.

        :param wl:      Wavelength of light for scattering calc (nm)
        :param Dp:      Diameter of particles (nm) or core diameter
        :param m:       Complex refractive index of particle
        :param nc:      Number of Legendre coefficients to compute. Auto if None.
        :param alt:     if True, will multiply legendre coeffs by (2*l+1)
        :return:        Legendre Coefficients as numpy array
        """
        # Get number of coeffs to calculate
        if nc is None:
            try:
                nc = self.mie._cache[self.mie._params_signature()]._coeffs.nmax
            except KeyError:
                spam = self.scat_phase_func(0., wl, Dp, m)  # analysis:ignore
                nc = self.mie._cache[self.mie._params_signature()]._coeffs.nmax
        # Calculate lth term Legendre coefficient for each term returned by Mie code
        lc = np.array([self._leg_coeff_calc(l, wl, Dp, m) for l in range(nc)])
        if alt:
            lc = np.array([lc[l]*(2*l+1) for l in range(nc)])
        return lc

    def spf_leg(self, theta, chi, alt=True):
        """Computes reconstructed scattering phase function from legendre coefficients.

        :param theta:   Angle of the phase function relative to incident light (deg)
        :param chi:     Legendre Coefficients
        :param alt:     if True, legendre coeffs are passed as chi_l * (2*l+1)
                        if False, legendre coeffs are just chi_l
        """
        x = np.cos(np.deg2rad(theta))
        # Calculate intensity at desired angle
        if np.isscalar(theta):
            if alt:
                return np.sum([chi[l]*self._leg_poly_calc(x,l) for l in range(len(chi))])
            return np.sum([(2*l+1)*chi[l]*self._leg_poly_calc(x,l) for l in range(len(chi))])
        else:
            if alt:
                return np.array([np.sum([chi[l]*self._leg_poly_calc(xx,l) for l in range(len(chi))]) for xx in x])
            return np.array([np.sum([(2*l+1)*chi[l]*self._leg_poly_calc(x,l) for l in range(len(chi))]) for xx in x])

    def _asy_calc(self, wl, Dp, m=np.complex(1.53, 0.0), DpC=None, mC=None):
        """Calculates the asymmetry parameter using Mie theory.
        Assumes spherical particles and cores/shells.
        NOTE: the imaginary component of the index of refraction (absorption) should be
        positive, e.g. opposite of actual value.
        Arguments:
            Dp:     Diameter of particles (nm) or shell diameter
            wl:     Wavelength of light for scattering calc (nm)
            m:      Complex refractive index or shell
            DpC:    Core Diameter of particles (nm) or None
            mC:     Core complex refractive index or None

        Returns:
            Asymmetry parameter
        """
        if np.isnan([wl, Dp]).any():
            return np.NaN
        # Single index of refraction
        if DpC is None:
            # Get Shape Parameter and Scattering Efficiency
            alpha = np.pi * Dp / wl
            self.mie._set_x(alpha)
            self.mie._set_m(m)
            return self.mie.asy()
        else:
            # Get Shape Parameter and Scattering Efficiency
            alpha1 = np.pi * Dp / wl
            alpha2 = np.pi * DpC / wl
            self.miecoat._set_y(alpha1)
            self.miecoat._set_x(alpha2)
            self.miecoat._set_m(mC)
            self.miecoat._set_m2(m)
            return self.miecoat.asy()

    def _ext_calc(self, N, wl, Dp, m=np.complex(1.53, 0.0), DpC=None, mC=None, ret_ext=True):
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
        """
        if np.isnan([N, wl, Dp]).any():
            return np.NaN
        # Single index of refraction
        if DpC is None:
            # Get Shape Parameter and Scattering Efficiency
            alpha = np.pi * Dp / wl
            self.mie._set_x(alpha)
            self.mie._set_m(m)
            qsca = self.mie.qsca()
            qabs = self.mie.qabs()
            # Calculate Scattering and Absorption Coefficients
            bsca = N * (np.pi/4.)*Dp**2 * qsca * 1e-6
            babs = N * (np.pi/4.)*Dp**2 * qabs * 1e-6
            if ret_ext:
                return (bsca + babs)
            return np.complex(bsca, babs)
        else:
            # Get Shape Parameter and Scattering Efficiency
            alpha1 = np.pi * Dp / wl
            alpha2 = np.pi * DpC / wl
            self.miecoat._set_y(alpha1)
            self.miecoat._set_x(alpha2)
            self.miecoat._set_m(mC)
            self.miecoat._set_m2(m)
            qsca = self.miecoat.qsca()
            qabs = self.miecoat.qabs()
            # Calculate Scattering and Absorption Coefficients
            bsca = N * (np.pi/4.)*Dp**2 * qsca * 1e-6
            babs = N * (np.pi/4.)*Dp**2 * qabs * 1e-6
            if ret_ext:
                return (bsca + babs)
            return np.complex(bsca, babs)

    def _ext_dist(self, dN, wl, Dp, gf=None, ind_mix=True, m=np.complex(1.53, 0.0),
                  DpC=None, DpCr=False, mC=None,
                  retdist=False, ret_ext=True):
        """Calculates extinction for an aerosol population using Mie theory.
        Uses particle number and diameter for a given size distribution.
        See self._ext_calc() docstring for more info.
            gf:         Growth factor for each Dp.
            ind_mix:    If True, will treat m as the dry index of refraction
                        and will adjust based on the dry and humidified diameters.
            retdist:    If True, returns the full distribution for each size,
                        otherwise returns total scattering for the distribution.
        """
        ext_arr = []
        n = len(dN)
        for i in range(n):
            if gf is None or gf is False:
                Dp_ = Dp[i]
            elif np.shape(gf) == np.shape(Dp):
                if gf[i] is None or gf[i] == False:
                    Dp_ = Dp[i]
                else:
                    Dp_ = Dp[i] * gf[i]
                    if ind_mix:
                        mmix = self._ind_mix(Dp[i], Dp_, m)
            else:
                raise Exception('gf must be the same shape as Dp array')
            if DpC is not None:
                if DpCr:
                    DpC_ = Dp[i] * DpC
                else:
                    DpC_ = DpC
            else:
                DpC_ = None
            if ind_mix:
                #ext_arr.append(self._ext_calc(dN[i], wl, Dp_, mmix, DpC_, mC))
                ext_arr.append(self._ext_calc(dN[i], wl, Dp_, np.complex(mmix.real,m.imag), DpC_, mC, ret_ext=ret_ext))
            else:
                ext_arr.append(self._ext_calc(dN[i], wl, Dp_, m, DpC_, mC, ret_ext=ret_ext))
        if retdist:
            return np.array(ext_arr)
        return np.sum(ext_arr)

    def ext_recon(self, varname='ExtRecon', wl=550.,
                  gf=False, ind_mix=True,
                  DpC=None, DpCr=False, mC=None,
                  bin_ind=True, low=None, up=None,
                  set_dist=False, save_gf=False, ret_ext=True):
        """Calculates the expected extinction coefficient of distribution in an optionally humidified environment for each datapoint.

        Can optionally only calculate extinction in a range of bin midpoint sizes using the low, up, and bin_ind parameters.
        If bin_ind is False, will find the lowest bin size included low for the lower index, and similar for up.
        Note: For now, as summing extinction across bins that aren't the same size requires 2D averaging and/or interpolation,
        it is not allowed. All CNdist instances must have same BinMid.

        Relative humidity and all aerosol properties, as well as multiple aerosol distributions, are all grabbed
        from self.props. Doesn't change these values within this method. Instantiate a new optical instance with
        different aerosol properties to change these.
        Note: Deliquescence and efflorescence are not accounted for currently, all particles are grown
        according to thermodynamic equilibrium using kappa-Kohler theory for any positive RH value (for gf=True).
        Caveat Emptor: For short atmospheric particle lifetimes, dust, or soot particles, etc., this may not be a
        particularly good assumption.

        Parameters:
            varname:    Name of variable to save
            wl:         Wavelength of light for scattering calc (nm)
            gf:         If True, will assume particles to be in equilibrium with RH.
                ind_mix:    If True, will use the volume mixing rule to adjust the real part of the index of
                            refraction for mixing with water (default m=1.33).
            # TODO: Update the DpC and DpCr variables to be contained in the aerosol properties of CNdist,
                    not passed here. All aerosol properties should be included in CNdist.
            DpC:        Diameter of the core, or None (assumed spherical)
            DpCr:       DpC ratio.
                            If True, will treat DpC as a multiplier to get core diameter by Dp_core = Dp_shell_dry*DpC
                            If False, DpC holds the (constant for full size range) diameter of the core.
            mC:         Core complex refractive index or None
            bin_ind:    If True, treats low and up as bin indicies, otherwise treats them as sizes for calc range
                low:        The lower bin index or size to use in scattering reconstruction. Default is min bin index.
                up:         The upper bin index or size to use in scattering reconstruction. Default is max bin index.
            CNdist:     A CNdist instance with the distribution data.
                        If None (Default), will use the referenced CNdist property in self (self.CNdist).
                        Otherwise, a reference to another CNdist object.
            set_dist:   If True, creates an additional db_ext/dlogDp extinction distribtion spectra variable,
                        in addition to the standard bext total extinction variable created by default.
                            Note: Extinction variables are summed across all modes, yielding
                            a 2D variable of shape [# dp, # bins] for the spectra variable.
                            Also, the db_ext/dlogDp variable must be divided by dlogDp as it is returned
                            as a db_ext variable for each bin.
            save_gf:    If True, will create and save a growth factor variable.
            ret_ext:    If True, will set variables as extinction.
                        If False, will set the variable as a complex number as np.complex(sca,abs)

        Saved variables:
            Will save the extinction variables for each aerosol population in self.pops,
            as well as a total in self._data.
        """
        # Check that bins are all the same #XXX: later can make this only needed if bin_ind is True
        #if bin_ind:
        for CNd in self.props.CNdist.__dict__.values():
            if (CNd.__dict__[self.data_obj_name].BinMid.d !=
                    self.props.CNdist.__dict__.values()[0].__dict__[self.data_obj_name].BinMid.d).any():
                #raise Exception('Bin indicies can not be used when CNdist have different BinMid')
                raise Exception('CNdist currently must all have the same BinMid')

        # Create optical reconstruction for each aerosol population
        for CNd in self.props.CNdist.__dict__.values():
            # Saves to self.pops.pop._data if more than one CNdist, or in self._data if only one
            if self.props.npop > 1:
                pop_obj = self.pops.__dict__[CNd.__name__]
            else:
                pop_obj = self
            ndp = CNd.ndp
            nm = CNd.nmodes
            # Get bin indices to integrate across
            if bin_ind is False:
                if low is None:
                    low = CNd.__dict__[self.data_obj_name].BinMid.d[0]
                if up is None:
                    up = CNd.__dict__[self.data_obj_name].BinMid.d[-1]
                tmp = np.where( (CNd.__dict__[self.data_obj_name].BinMid.d >= low) &
                                (CNd.__dict__[self.data_obj_name].BinMid.d <= up) )[0]
                low = tmp[0]
                up = tmp[-1]+1
            else:
                if low is None:
                    low = 0
                if up is None:
                    up = CNd.__dict__[self.data_obj_name].BinMid.d.size
            BinMid = CNd.__dict__[self.data_obj_name].BinMid.d[low:up]
            nbins = BinMid.size
            # For wet particle calculation, growth factor calculated from RH and kappa for each mode
            gf_ = np.zeros([ndp,nm,nbins])
            gf_[:] = np.NaN
            if gf is True:
                try:
                    kappa_ = CNd.__dict__[self.data_obj_name].kappa.d
                except AttributeError:
                    kappa_ = self.__dict__[self.data_obj_name].kappa.d
                # Calculate growth factor at each datapoint, mode, and dry diameter
                for ind in range(ndp):
                    for m in range(nm):
                        gf_[ind,m] = np.array([self.parent.kappa.gf(self.props.RH.d[ind], Dd, kappa_[ind,m]) for Dd in BinMid])
            else:
                # No growth factor
                gf_ = np.zeros([ndp,nm,nbins], dtype=bool)
            # Get index of refraction for each mode and datapoint
            dry_m_ = np.zeros([ndp,nm], dtype=np.complex)     #,nbins])
            for ind in range(ndp):
                for m in range(nm):
                    try:
                        dry_m_[ind,m] = CNd.__dict__[self.data_obj_name].dry_m.d[ind,m]
                    except AttributeError:
                        dry_m_[ind,m] = self.__dict__[self.data_obj_name].dry_m.d[ind,m]

            # Calculate extinction for each datapoint
            if ret_ext:
                ext = np.zeros([ndp,nm,nbins])          # will have dimensions [# dp, # modes, # bins]
            else:
                ext = np.zeros([ndp,nm,nbins], dtype=np.complex)
            for ind in range(ndp):
                # Calculate each mode separately
                for m in range(nm):
                    if nm == 1:
                        # If only one mode, don't grab from CNdist.modes
                        dN = CNd.__dict__[self.data_obj_name].dN.d[ind]
                    else:
                        # If more than one mode, grab all dN variables from CNdist.modes
                        dN = CNd.modes.__dict__['m%i'%(m+1)].__dict__[self.data_obj_name].dN.d[ind]
                    ext[ind,m] = self._ext_dist(dN[low:up], wl, BinMid,
                                                gf_[ind,m], ind_mix, dry_m_[ind,m],
                                                DpC, DpCr, mC, True, ret_ext)

            # Create and set needed variables
            self._make_var(varname, def_varname='bext', data_obj=pop_obj._data)
            if set_dist:
                self._make_var(varname+'_dist', def_varname='dbextdlogDp', data_obj=pop_obj._data)
            val = np.sum(ext, axis=(1,2))
            self._set_var(varname, val, data_obj=pop_obj._data)
            if set_dist:
                # db_ext divided by dlogDp to get db_ext/dlogDp spectra
                val = np.sum(ext, axis=1) / CNd.__dict__[self.data_obj_name].dlogDp.d[low:up]
                self._set_var(varname+'_dist', val, data_obj=pop_obj._data)
            if save_gf:
                self._make_var('gf', data_obj=pop_obj._data)
                self._set_var('gf', gf_, data_obj=pop_obj._data)

        # Sum extinction for all CNdist if more than one population CNdist
        if self.props.npop > 1:
            self._make_var(varname, def_varname='bext', data_obj=self._data)
            if set_dist:
                self._make_var(varname+'_dist', def_varname='dbextdlogDp', data_obj=self._data)
            val = np.sum([po._data.__dict__[varname].d for po in self.pops.__dict__.values()], axis=0)
            self._set_var(varname, val, data_obj=self._data)
            if set_dist:
                # db_ext divided by dlogDp to get db_ext/dlogDp spectra
                val = np.sum([po._data.__dict__[varname+'_dist'].d for po in self.pops.__dict__.values()], axis=0)
                self._set_var(varname+'_dist', val, data_obj=self._data)

    def norm_dbextdlogDp(self, varname='Norm_dbextdlogDp', v=None, data_obj=None, altdbextkey=None, norm_vals=None, add_dbext=False):
        """Creates a normalized dVdlogDp variable.
        Requires self.data.dbextdlogDp and self.data.dbext.
        If norm_vals is given, it contains an array of values to normalize against. Otherwise
        will use the data_obj.bext.d[i] value. Alternately, can contain a string to grab an
        array from data_obj.__dict__[norm_vals].d
        #XXX: add_dbext: if true, will add a 'Norm_dbext' variable as well. #TODO: dif varname if str?
        """
        if data_obj is None:
            data_obj = self.data
        if altdbextkey is None:
            dbextdlogDp = data_obj.dbextdlogDp
        else:
            dbextdlogDp = data_obj.__dict__[altdbextkey]
        if norm_vals is None:
            if not hasattr(data_obj, 'bext'):
                raise Exception
            norm_vals = data_obj.bext.d
            nv = True
        else:
            if isinstance(norm_vals, str):
                norm_vals = data_obj.__dict__[norm_vals].d
            if not isinstance(norm_vals, np.ndarray):
                raise Exception
            if norm_vals.size != dbextdlogDp.size:
                raise Exception
            nv = False
        norm_dbextdlogDp = []
        for i in range(dbextdlogDp.size):
            norm_spec = []
            for j in range(dbextdlogDp.length):
                norm_spec.append(dbextdlogDp.d[i,j]/norm_vals[i])
            norm_dbextdlogDp.append(norm_spec)
        norm_dbextdlogDp = np.array(norm_dbextdlogDp)
        # Add the Norm_dV variable if needed
        if add_dbext:
            norm_dbext = norm_dbextdlogDp * data_obj.dlogDp.d
        # v is the same as dNdlogDp as long as CNconc is valid as well
        if v is None:
            v = copy.copy(dbextdlogDp.v)
            if nv:
                for i in range(data_obj.bext.size):
                    if not data_obj.bext.v[i]:
                        v[i] = [False]*dbextdlogDp.length
        self._make_var(varname, def_varname='Norm_dbextdlogDp', data_obj=data_obj)
        self._set_var(varname, norm_dbextdlogDp, v=v, vec=dbextdlogDp.vec, data_obj=data_obj)
        if add_dbext:
            self._make_var('Norm_dbext', data_obj=data_obj)
            self._set_var('Norm_dbext', norm_dbext, v=v, vec=dbextdlogDp.vec, data_obj=data_obj)

    def _path_AOD(self, ext_var, ind, path_length):
        """Returns the optical depth along a path using an extinction coefficient variable.
        Parameters:
            ext_var:        The extinction coefficient variable object for each datapoint in ind (Mm^-1).
            ind:            The ordered indices of the path to calculate extinction over.
            path_length:    The path length associated with each datapoint in ind (m).
        """
        # Check that path_length is the same size as ind
        if len(ind) != len(path_length):
            raise Exception('Size of ind and path_length not equal')
        # Extinction (Aerosol Optical Depth) for each datapoint
        AOD_grid = ext_var.d[ind] * path_length / 1e6
        AOD_cumgrid = np.cumsum(AOD_grid)
        AOD = np.sum(AOD_grid)
        AOD_frac = AOD_grid / AOD
        AOD_cumfrac = np.cumsum(AOD_frac)
        return (AOD, AOD_grid, AOD_cumgrid, AOD_frac, AOD_cumfrac)

    def _grid_edges(self, loc):
        """Calculates bin edges at midpoint between grid centers for loc.
        """
        gpl = self._grid_path_length(loc)
        return np.concatenate((loc-gpl/2., [loc[-1]+gpl[-1]/2.]))

    def _grid_path_length(self, loc):
        """Calculates the length of assumed adjacent grid boxes defined at loc..
        Note that loc must be defined in a constant metric (e.g. meters, rather than lat or lon position).
        Returns the length of each grid box in the same units loc is defined in.
        """
        grid_sep_dist = np.abs(loc[1:] - loc[:-1])
        grid_path_length = np.array([grid_sep_dist[0]] +
                                    [grid_sep_dist[i]/2. + grid_sep_dist[i+1]/2. for i in range(grid_sep_dist.size-1)] +
                                    [grid_sep_dist[-1]])
        return grid_path_length

    def path_AOD(self, AODname='AODRecon', ext_var=None, pop_obj=None,
                 loc_center=True, loc_var=None, loc_low=None, loc_up=None, ind=None):
        """Calculates the optical depth along a path using an extinction coefficient variable.
        The path is defined by ind, which is assumed to be a series of touching grid boxes, with
        locations defined in the loc_var variable.
        Creates a data object for each calculation called AODname that holds five AOD variables.
        Parameters:
            AODname:        The name of the AOD variable to be created.
            ext_var:        The extinction coefficient variable for each datapoint in ind (Mm^-1).
                            Can also include a string of the variable name as self.data.ext_var.
                            Typically should pass as a string in case multiple aerosol populations
                            are used.
            pop_obj:        If AOD is to be calculated for only one aerosol population CNdist object,
                            set that object here. If None, will calculate a total AOD using self and
                            an AOD for each population in self.pops.
            loc_center:     If True, will calculate path lengths for indices at a location center point.
                            Path lengths will be calculated based on length between points.
                            If False, will expect bin edges associated with each index.
            loc_var:        The location variable with location for each datapoint in ind in meters (m).
                            Can also be a string of the variable name in self.data.
            loc_low:        If loc_center is False, will use this as the lower edge for each datapoint (m).
                            Can also be a string of the variable name in self.data.
            loc_up:         If loc_center is False, will use this as the upper edge for each datapoint (m).
                            Can also be a string of the variable name in self.data.
            ind:            The ordered indices of the path to calculate extinction over.
                            If 2D array, will calculate for each path_ind in the array.
                            This will result in AOD array variables for each path in ind.
        Saved Variables:
            AOD:            The total AOD along the path.
            AOD_grid:       The AOD associated with each grid datapoint.
            AOD_cumgrid:    The cumulative AOD associated with each grid datapoint.
            AOD_frac:       The fraction of the total AOD in each grid datapoint.
            AOD_cumfrac:    The cumulative fraction of the total AOD in each grid datapoint along the path.
        """
        # Get population object that has a ._data attribute
        if pop_obj is None:
            pop_obj_arr = [self]
            if hasattr(self, 'pops'):
                pop_obj_arr = pop_obj_arr + self.pops.__dict__.values()
        else:
            pop_obj_arr = [pop_obj]

        # Set up grid location variables
        if loc_center:
            # Check if loc_var is a string with variable
            if isinstance(loc_var, str):
                loc_var = self._data.__dict__[loc_var]
            else:
                self._make_var('loc_var', def_varname='loc', data_obj=self._data)
                self._set_var('loc_var', np.atleast_1d(loc_var), data_obj=self._data)
                loc_var = self._data.loc_var
        else:
            if isinstance(loc_low, str):
                loc_low = self._data.__dict__[loc_low]
            else:
                self._make_var('loc_low', def_varname='loc', data_obj=self._data)
                self._set_var('loc_low', np.atleast_1d(loc_low), data_obj=self._data)
                loc_low = self._data.loc_low
            if isinstance(loc_up, str):
                loc_up = self._data.__dict__[loc_up]
            else:
                self._make_var('loc_up', def_varname='loc', data_obj=self._data)
                self._set_var('loc_up', np.atleast_1d(loc_up), data_obj=self._data)
                loc_up = self._data.loc_up

        # If ind is None, assume all datapoints are used and already ordered.
        if ind is None:
            ind = np.array(range(ext_var.d.size))
        # Otherwise, determine if an array of AOD calcs for various paths is needed
        else:
            ind = np.atleast_1d(ind)
            if ind.ndim == 1:
                mult_calc = False
            elif ind.ndim == 2:
                mult_calc = ind.shape[0]
            else:
                raise Exception

        # AOD calcs for each pop object and total
        for pop_obj in pop_obj_arr:
            # Multiple AOD path calculations
            if mult_calc:
                path_dp = ind.shape[1]
                # Get grid path lengths
                if loc_center:
                    path_lengths = np.array([self._grid_path_length(loc_var.d[ind[i]]) for i in range(mult_calc)])
                else:
                    path_lengths = np.array([loc_up.d[ind[i]] - loc_low.d[ind[i]] for i in range(mult_calc)])
                # Grab ext_var from self if needed
                if isinstance(ext_var, str):
                    ext_var = pop_obj._data.__dict__[ext_var]
                # Create variables
                AOD = np.zeros(mult_calc)
                AOD_grid = np.zeros([mult_calc,path_dp])
                AOD_cumgrid = np.zeros([mult_calc,path_dp])
                AOD_frac = np.zeros([mult_calc,path_dp])
                AOD_cumfrac = np.zeros([mult_calc,path_dp])
                for i in range(mult_calc):
                    AOD_, AOD_grid_, AOD_cumgrid_, AOD_frac_, AOD_cumfrac_ = self._path_AOD(ext_var, ind[i], path_lengths[i])
                    AOD[i] = AOD_
                    AOD_grid[i] = AOD_grid_
                    AOD_cumgrid[i] = AOD_cumgrid_
                    AOD_frac[i] = AOD_frac_
                    AOD_cumfrac[i] = AOD_cumfrac_

            # Single AOD path calculation
            else:
                # Get grid path lengths
                if loc_center:
                    path_lengths = self._grid_path_length(loc_var.d[ind])
                else:
                    path_lengths = loc_up.d[ind] - loc_low.d[ind]
                # Grab ext_var from self if needed
                if isinstance(ext_var, str):
                    ext_var = pop_obj.data.__dict__[ext_var]
                # Call function to set variables
                AOD, AOD_grid, AOD_cumgrid, AOD_frac, AOD_cumfrac = self._path_AOD(ext_var, ind, path_lengths)

            # Create path_AOD data object and associated variables
            setattr(pop_obj._data, AODname, BlankObject())
            do = pop_obj._data.__dict__[AODname]
            self._make_var(varname='AOD', data_obj=do)
            self._make_var(varname='AOD_grid', data_obj=do)
            self._make_var(varname='AOD_cumgrid', data_obj=do)
            self._make_var(varname='AOD_frac', data_obj=do)
            self._make_var(varname='AOD_cumfrac', data_obj=do)
            self._set_var('AOD', AOD, data_obj=do)
            self._set_var('AOD_grid', AOD_grid, data_obj=do)
            self._set_var('AOD_cumgrid', AOD_cumgrid, data_obj=do)
            self._set_var('AOD_frac', AOD_frac, data_obj=do)
            self._set_var('AOD_cumfrac', AOD_cumfrac, data_obj=do)

    def _bext_plot(self, wl, m, Dpl, Dpu, nbins=50):
        """Plots the extinction coefficient across a range of particle sizes.
        This was mostly for testing purposes. Probably not really needed.
        """
        BinMid = np.logspace(np.log10(Dpl),np.log10(Dpu),nbins)
        bext = np.array([self._ext_calc(1, wl, Dp, m=m) for Dp in BinMid])
        pl.figure()
        pl.plot(BinMid, bext)
        pl.xlim(Dpl, Dpu)


# --- Combined optical analysis parent class ---
class OptAnalysis(BaseAnalysis):
    """A class containing needed methods and class instances to conduct an optical reconstruction of an aerosol population.
    """
    def __init__(self, var_dir=None, gvar={}, var={}, **kwargs):
        """
        Parameters:
            var_dir:    A directory to save files in. If None, current working directory.
            gvar:       Dictionary of global or metadata variables
            var:        Dictionary of varibles needed in current analysis
        """
        # Set initial number of datapoints in current analysis to None
        self.n = None
        # Set variable or file storage directory
        if var_dir is None:
            self._var_dir = os.getcwd()
        # Needed analysis classes
        self.kappa = KappaCalc(self, self._var_dir)
        # Call BaseAnalysis Class init method to finish import
        BaseAnalysis.__init__(self, None, None)
        # Create default data attribute and setup variables
        setattr(self, 'data', BlankObject())
        self._setup_dataobj(self.data, gvar, var)
        # Default CNdist and Optical instance storage points for method delegation
        self._cndef = None
        self._optdef = None
        # Check for external objects to be assigned as attributes
        if 'CNdist' in kwargs:
            spam = kwargs['CNdist']
            if type(spam) is list or type(spam) is np.ndarray:
                for cnd in spam:
                    setattr(self, cnd.__name__, cnd)
                if self._cndef is None:
                    self._cndef = self.__dict__[spam[0].__name__]
            else:
                setattr(self, spam.__name__, spam)
                if self._cndef is None:
                    self._cndef = self.__dict__[spam.__name__]

    def cn_dist(self, **kwargs):
        """Explicit delegation of a new CNdist() class instance as an attribute of self.
        """
        # Create new CNdist object that generates a new parameterized aerosol population
        kwargs.update(gen_dist=True)
        CNdist(self, **kwargs)
        # Check that the new CNdist instance does not have a different number of data points than self.n
        # Update self.n to have the number of datapoints in the newly created CNdist instance
        if 'name_ovr' in kwargs:
            if kwargs['name_ovr'] is not None:
                name = kwargs['name_ovr']
        else:
            name = 'cn'
        if self.n is not None:
            if self.__dict__[name].ndp != self.n:
                print 'Warning: New number of datapoints in CNdist ({}) not equal to existing number ({})'.format(self.__dict__[name].ndp, self.n)
                print 'Requested CNdist instance removed.'
                delattr(self, name)
        else:
            self.n = self.__dict__[name].ndp
        # Fill default CNdist instance if needed
        if self._cndef is None:
            self._cndef = self.__dict__[name]
        elif self._cndef.__name__ != 'cn' and name == 'cn':
            self._cndef = self.cn

    def _mk_cndef(self):
        """Creates a dummy CNdist instance.
        """
        self.cn_dist(num_parm=np.array([100.,1.5,1.0]), CNconc=1., modes=1, auto_bins=True, nbins=3)

    def optical(self, cn_dist=None, rh=None, **kwargs):
        """Explicit delegation of a new Optical() class instanceas an attribute of self.
        """
        # Creation of Optical instance requires an existing CNdist instance
        if self._cndef is None:
            self._mk_cndef()
        if cn_dist is None:
            cn_dist = self._cndef
        if rh is None:
            if not 'RH' in kwargs:
                rh = [0.]*self.n
                kwargs.update(var={'RH':rh})
        # Create new Optical object that allows for an optical reconstruction
        Optical(self, CNdist=cn_dist, **kwargs)
        # Check that the new CNdist instance does not have a different number of data points than self.n
        # Update self.n to have the number of datapoints in the newly created CNdist instance
        if 'name_ovr' in kwargs:
            if kwargs['name_ovr'] is not None:
                name = kwargs['name_ovr']
        else:
            name = 'opt'
        if self.n is not None:
            if self.__dict__[name].data.RH.d.size != self.n:
                print 'Warning: New number of datapoints in Optical ({}) not equal to existing number ({})'.format(self.__dict__[name].data.RH.d.size, self.n)
                print 'Requested Optical instance removed.'
                delattr(self, name)
        else:
            self.n = self.__dict__[name].ndp
        # Fill default Optical instance if needed
        if self._optdef is None:
            self._optdef = self.__dict__[name]
        elif self._optdef.__name__ != 'opt' and name == 'opt':
            self._optdef = self.opt

    def opt_props(self, cn_dist, RH, opt_inst='opt'):
        """Resets the properties of an Optical class for the given CNdist populations and RH values.
        The default Optical() class instance of self.opt will be used unless opt_inst is passed.
        """
        self.__dict__[opt_inst]._set_props(cn_dist, RH)

    def ext_calc(self, N, wl, Dp, m=np.complex(1.53, 0.0), DpC=None, mC=None, ret_ext=True):
        """Calculates extinction due to scattering and absorption using Mie theory.
        Pass through method for access to Optical._ext_calc().
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
        # Check if the needed default CNdist and Optical class instances extists and instantiates it if not
        if self._cndef is None:
            self._mk_cndef()
        if self._optdef is None:
            self.optical()
        # Explicit delgation of Optical._ext_calc()
        return self._optdef._ext_calc(N, wl, Dp, m=m, DpC=DpC, mC=mC, ret_ext=ret_ext)

    def ssa_calc(self, wl, Dp, m=np.complex(1.53, 0.0), DpC=None, mC=None):
        """Calculates particle single scatter albedo using Mie theory.
        Uses the self.ext_calc() result to get scattering and absorption coefficients
        and uses these to calculate single scatter albedo for a single particle.
        See self.ext_calc() docstring for parameter information.
        Returns:
            Single scatter albedo
        """
        spam = self.ext_calc(1., wl, Dp, m=m, DpC=DpC, mC=mC, ret_ext=False)
        return spam.real / (spam.real+spam.imag)

    def asy_calc(self, wl, Dp, m=np.complex(1.53, 0.0), DpC=None, mC=None):
        """Calculates particle asymmetry parameter using Mie theory.
        See self.ext_calc() docstring for parameter information.
        Returns:
            Asymmetry parameter
        Note: Explicit delegation of method from Optical() class.
        """
        # Check if the needed default CNdist and Optical class instances extists and instantiates it if not
        if self._cndef is None:
            self._mk_cndef()
        if self._optdef is None:
            self.optical()

        return self._optdef._asy_calc(wl, Dp, m=m, DpC=DpC, mC=mC)

    def S12_calc(self, theta, wl, Dp, m=np.complex(1.53, 0.0), DpC=None, mC=None):
        """Calculates particle Mie theory amplitude scattering matrix components S1 and S2.
        See self.ext_calc() docstring for parameter information.
        Additional Parameters:
            theta:  Scattering angle (deg)
        Returns:
            tuple of (S1, S2)
        Note: Explicit delegation of method from Optical() class.
        """
        # Check if the needed default CNdist and Optical class instances extists and instantiates it if not
        if self._cndef is None:
            self._mk_cndef()
        if self._optdef is None:
            self.optical()

        return self._optdef._S12_calc(wl, Dp, costh, m=m, DpC=DpC, mC=mC)

    def spf(self, theta, wl, Dp, m=np.complex(1.53, 0.0), DpC=None, mC=None):
        """Calculates the single particle scattering phase function.
        See self.ext_calc() docstring for parameter information.
        Additional Parameters:
            theta:  Scattering angle (deg) [list or array for multiple angles]
        Returns:
            Scattering phase function at angle theta
        Note: Explicit delegation of method from Optical() class.
              Core/shell option not setup yet.
        """
        # Check if the needed default CNdist and Optical class instances extists and instantiates it if not
        if self._cndef is None:
            self._mk_cndef()
        if self._optdef is None:
            self.optical()

        return self._optdef.scat_phase_func(theta, wl, Dp, m)

    def legendre_coeffs(self, wl, Dp, m, nc=None, alt=True):
        """Calculates the Legendre coefficients for the phase function.
        See self.ext_calc() docstring for parameter information.
        Additional Parameters:
            nc:     Number of Legendre coefficients to compute. Auto if None.
            alt:    if True, will multiply legendre coeffs by (2*l+1)
        Returns:
            Array of Legendre coefficients
        Note: Explicit delegation of method from Optical() class.
              Core/shell option not setup yet.
        """
        # Check if the needed default CNdist and Optical class instances extists and instantiates it if not
        if self._cndef is None:
            self._mk_cndef()
        if self._optdef is None:
            self.optical()

        return self._optdef.legendre_coeffs(wl, Dp, m, nc, alt)

    def spf_leg(self, theta, chi, alt=True):
        """Computes reconstructed scattering phase function from legendre coefficients.
        Parameters:
            theta:  Angle of the phase function relative to incident light (deg)
            chi:    Legendre Coefficients
            alt:    if True, legendre coeffs are passed as chi_l * (2*l+1)
                    if False, legendre coeffs are just chi_l
        Returns:
            Scattering phase function at angle theta from legendre reconstruction
        Note: Explicit delegation of method from Optical() class.
              Core/shell option not setup yet.
        """
        # Check if the needed default CNdist and Optical class instances extists and instantiates it if not
        if self._cndef is None:
            self._mk_cndef()
        if self._optdef is None:
            self.optical()

        return self._optdef.spf_leg(theta, chi, alt)

    # TODO: fix order of wl, Dp, m in methods to be consistent
