#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as scipy
import scipy.integrate as integrate
import scipy.special

if __name__ == "__main__":

    Umax = 101
    dU = 0.1
    for intU in range(Umax):
        U = dU*intU
# f[U_] = -4 Integrate[ BesselJ[0, x] BesselJ[1, x]/x/(1 + Exp[0.5 U*x]), {x, 0, Infinity}]
        ene = integrate.quad(lambda x, c : \
            -4.0 * scipy.special.j0(x) * scipy.special.j1(x) \
            / x / (1+np.exp(0.5*c*x)) \
            , 0.0, np.inf, args=U \
            , epsabs=1e-11, epsrel=1e-11, limit=10000)
        print("{0:.16f} {1:.16f} {2:.16f}".format(U,ene[0],ene[1]))

    print("# exact (4/pi):",0.0,4.0/np.pi)
