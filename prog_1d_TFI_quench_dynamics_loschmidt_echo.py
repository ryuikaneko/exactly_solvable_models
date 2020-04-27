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
    parser.add_argument('-tau',metavar='tau',dest='tau',type=np.float,default=5.0,help='max time: tau=vmax*tmax')
    parser.add_argument('-dt',metavar='dt',dest='dt',type=np.float,default=0.01,help='step size: dt')
    return parser.parse_args()

def epsion(J,h,k):
    return 2.0*J*np.sqrt(1.0+h**2-2.0*h*np.cos(k))

def diff_log_loschmidt_echo(J,h0,h1,k,t):
    e0 = epsion(J,h0,k)
    e1 = epsion(J,h1,k)
    denom = e1 * e0
    numer = 4.0*J**2 * (h1-h0) * np.sin(k) * np.sin(e1*t)
    return np.log(1.0 - (numer/denom)**2) / (-2.0*np.pi)

def diff_log_loschmidt_echo_main(k,c):
    return diff_log_loschmidt_echo(c[0],c[1],c[2],k,c[3])

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
    list_log_loschmidt_echo = []
    vmax = 2.0*J*np.min([1.0,h1])
    tmax = tau/vmax
    N = int(tmax/dt+0.1)+1
    for steps in range(N):
        t = steps*dt
        log_loschmidt_echo = integrate.quad(lambda k,c : \
            diff_log_loschmidt_echo_main(k,c), 0.0, np.pi, args=[J,h0,h1,t])
        print("{0:.16f} {1:.16f} {2:.16f} {3:.16f}".format(t,vmax*t,log_loschmidt_echo[0],log_loschmidt_echo[1]))
#        list_t.append(t)
        list_vt.append(vmax*t)
        list_log_loschmidt_echo.append(log_loschmidt_echo[0])
#
    fig1 = plt.figure()
    fig1.suptitle("")
    plt.plot(list_vt,list_log_loschmidt_echo)
    plt.xlabel("$v_{max}t$")
    plt.ylabel("log Loschmidt echo")
    fig1.savefig("fig_1d_TFI_quench_dynamics_loschmidt_echo.png")

if __name__ == "__main__":
    main()
