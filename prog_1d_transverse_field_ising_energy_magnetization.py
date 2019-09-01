#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as scipy
import scipy.integrate as integrate

if __name__ == "__main__":

#  gmax = 151
#  dg = 0.01
  gmax = 16
  dg = 0.1

  for intg in range(gmax):
    g = dg*intg

    # exact energy
    f = lambda x,c : -np.sqrt(1.0+c**2-2.0*c*np.cos(x))/np.pi
    E_exact = integrate.quad(f,0,np.pi,args=g)[0]

    # exact magnetization
    if g==0:
      Mx_exact = 0.0
    elif g==1:
      Mx_exact = 2.0/np.pi
    else:
      g1 = lambda x,c : 1.0/np.sqrt(1.0+c**2+2.0*c*np.cos(x))/np.pi
      g2 = lambda x,c : 1.0/np.sqrt(1.0+c**2+2.0*c*np.cos(x))*np.cos(x)/np.pi
      Mx_exact = integrate.quad(g1,0,np.pi,args=1.0/g)[0] \
        + integrate.quad(g2,0,np.pi,args=1.0/g)[0] / g

    print("{0:.16f} {1:.16f} {2:.16f}".format(g,E_exact,Mx_exact))
