#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as scipy
import scipy.integrate as integrate
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Dynamics of S=1/2 TFI chain')
    parser.add_argument('-J',metavar='J',dest='J',type=np.float,default=1.0,help='interaction: J')
#    parser.add_argument('-h0',metavar='h0',dest='h0',type=np.float,default=0.0,help='initial field: h0')
#    parser.add_argument('-h1',metavar='h1',dest='h1',type=np.float,default=0.5,help='final field: h1')
    parser.add_argument('-h0',metavar='h0',dest='h0',type=np.float,default=0.6,help='initial field: h0')
    parser.add_argument('-h1',metavar='h1',dest='h1',type=np.float,default=0.3,help='final field: h1')
#    parser.add_argument('-tau',metavar='tau',dest='tau',type=np.float,default=10.0,help='max time: tau=vmax*tmax')
    parser.add_argument('-tau',metavar='tau',dest='tau',type=np.float,default=20.0,help='max time: tau=vmax*tmax')
    parser.add_argument('-dt',metavar='dt',dest='dt',type=np.float,default=0.01,help='step size: dt')
    return parser.parse_args()

def epsion(J,h,k):
    return 2.0*J*np.sqrt(1.0+h**2-2.0*h*np.cos(k))

def diff_m_ising(J,h0,h1,k):
    e0 = epsion(J,h0,k)
    e1 = epsion(J,h1,k)
    e1prime = 4.0*J**2*h1*np.sin(k)/e1
    cosDk = 4.0*J**2*(h1*h0-(h1+h0)*np.cos(k)+1.0)/e1/e0
    return e1prime*np.log(np.abs(cosDk))/np.pi

def diff_m_ising_main(k,c):
    return diff_m_ising(c[0],c[1],c[2],k)

def calc_CFF(h0,h1):
    assert np.abs(h0)<=1.0 and np.abs(h1)<=1.0, "!!! choose |h0|,|h1|<1"
    numer = 1.0-h1*h0+np.sqrt((1.0-h1**2)*(1.0-h0**2))
    denom = 2.0*np.sqrt((1.0-h1*h0)*np.sqrt(1.0-h0**2))
    return np.sqrt(numer/denom)

def main():
    args = parse_args()
    J = args.J
    h0 = args.h0
    h1 = args.h1
    tau = args.tau
    dt = args.dt
#
#    list_t = []
    list_vt = []
    list_m_ising = []
    vmax = 2.0*J*np.min([1.0,h1])
    tmax = tau/vmax
    N = int(tmax/dt+0.1)+1
#
    CFF = calc_CFF(h0,h1)
    integral_m_ising = integrate.quad(lambda k,c : \
        diff_m_ising_main(k,c), 0.0, np.pi, args=[J,h0,h1])
    print("# m(t)~a*exp(bt)")
    print("# a~ {0:.16f}".format(CFF))
    print("# b~ {0:.16f} {1:.16f}".format(integral_m_ising[0],integral_m_ising[1]))
#
    for steps in range(N):
        t = steps*dt
        m_ising = CFF*np.exp(t*integral_m_ising[0])
        print("{0:.16f} {1:.16f} {2:.16f}".format(t,vmax*t,m_ising))
#        list_t.append(t)
        list_vt.append(vmax*t)
        list_m_ising.append(m_ising)
#
    fig1 = plt.figure()
    fig1.suptitle("")
    plt.plot(list_vt,list_m_ising)
    plt.xlabel("$v_{max}t$")
    plt.ylabel("$m_{Ising}$")
    fig1.savefig("fig_1d_TFI_quench_dynamics_m_ising.png")

if __name__ == "__main__":
    main()
