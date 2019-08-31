#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as scipy
import scipy.integrate as integrate

if __name__ == "__main__":
    N = 100
    for icoseta in range(N):
        coseta = icoseta*1.0/N
        eta = np.arccos(coseta)
        val = integrate.quad(lambda x, c : \
            2*np.exp(-2*c*x) \
            * (1-np.exp(-2*(np.pi-c)*x)) \
            / (1-np.exp(-2*np.pi*x)) \
            / (1+np.exp(-2*c*x)) \
            , 0.0, np.inf, args=eta)
        print (coseta,0.25*coseta-val[0]*np.sin(eta))

    print ("# exact (-1/pi):",0.0,-1.0/np.pi)
    print ("# exact (1/4-ln2):",1.0,0.25-np.log(2.0))
