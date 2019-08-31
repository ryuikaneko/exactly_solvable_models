#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as scipy
import scipy.integrate as integrate

if __name__ == "__main__":

## set anisotropy g

#  gmax = 151
#  dg = 0.01
#  for intg in range(gmax):
#    g = dg*intg - 0.5

  arr_g = np.arange(1.00,0.25,-0.1)
  arr_g = np.append(arr_g,np.arange(0.25,0.01,-0.05))
  arr_g = np.append(arr_g,np.arange(0.00,-0.49,-0.05))
  arr_g = np.append(arr_g,-0.5)
  for g in arr_g:

## c: anisotropy
##  c=1: isolated dimers
##  c=0: isotropic
##  c=-0.5: decoupled chains

#    eps_k = lambda x,y,c : (1.0+2.0*c) + (1.0-c)*np.cos(x) + (1.0-c)*np.cos(y) 
#    del_k = lambda x,y,c : (1.0-c)*np.sin(x) - (1.0-c)*np.sin(y)
#    f = lambda x,y,c : np.sqrt(eps_k(x,y,c)**2 + del_k(x,y,c)**2)

    f = lambda x,y,c : np.sqrt(
      + (1.0-c)*(1.0-c) + (1.0-c)*(1.0-c) + (1.0+2.0*c)*(1.0+2.0*c)
      + 2.0*(1.0-c)*(1.0-c)*np.cos(x-y)
      + 2.0*(1.0-c)*(1.0+2.0*c)*np.cos(x)
      + 2.0*(1.0-c)*(1.0+2.0*c)*np.cos(y)
      )

## integration
##   use `nquad' for better accuracy

#    E_exact = - np.array( integrate.dblquad(f,-np.pi,np.pi,
#      lambda x:-np.pi,lambda x:np.pi,
#      args=(g,),epsrel=1e-10) ) / (16.0*np.pi*np.pi) / 2.0

    options={'limit':1000,'epsrel':1e-11,'epsabs':1e-11}
    E_exact = - np.array( integrate.nquad(f,[[-np.pi,np.pi],
      [-np.pi,np.pi]],args=(g,),
      opts=[options,options]
      ) ) / (16.0*np.pi*np.pi) / 2.0

    print("{0:.16f} {1:.16f} {2:.16f}".format(g,E_exact[0],E_exact[1]))
