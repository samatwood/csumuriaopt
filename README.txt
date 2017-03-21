# csumuriaopt.py

###
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
###


See examples.py for example uses of this code in via python scripts or in
python interactive mode.