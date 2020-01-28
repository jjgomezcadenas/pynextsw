import numpy as np
import pandas as pd
import os, sys
from scipy import signal
from  . system_of_units import *


def delta_signal(pre=10, peak=2, tail=100):
    return np.concatenate((np.zeros(pre),np.ones(peak),np.zeros(tail)))


def shaper(signal_in, r, c, f_sample=1e+9):
    freq_LPF = 1/(r*c)
    freq_LPFd = freq_LPF / (f_sample*np.pi); # Normalized by Nyquist Freq (half-cycles/sample)
    b, a = signal.butter(1, freq_LPFd, 'low', analog=False)
    signal_out = signal.lfilter(b,a,signal_in)
    return signal_out


def scint_photons(energy, Ws =39.2 * eV):
    return energy / Ws


def ionisation_electrons(energy, Wi =21.9 * eV):
    return energy / Wi


def el_photons(energy, Wi =21.9 * eV, EL = 500):
    return EL * ionisation_electrons(energy, Wi)


def detector_perimeter(diameter):
    return 2 * np.pi * diameter / 2.


def detector_sensitive_area(d=2.5*m, coverage=1.0):
    return coverage * np.pi * (d/2)**2


def detector_total_DCR(area, dcrPerUnitArea):
    return area * dcrPerUnitArea


def n_sipms(areaDetector, areaSiPMs, coverage=0.9):
    return int(coverage * areaDetector / areaSiPMs)


def n_fibers (detectorD, fiberD):
    return detector_perimeter(detectorD) / fiberD


def scint_photons_fiber (energy, effF, effS, effT):
    return scint_photons(energy) * effF *  effS * effT


def el_photons_fiber (energy, effF, effS, effT):
    return el_photons(energy) * effF *  effS * effT


def dcr_sipm_per_unit_area (tC, TK, F=0.66):
    tK = tC + 273.15
    ldcr = TK(1/tK)
    return F* np.exp(ldcr) * hertz / mm2


def dcr_sipm_per_time (tC, TK, sipmX, time, F=0.66):
    return (sipmX**2/mm2) * (time/second) * dcr_sipm_per_second_per_mm2 (tC, TK, F)


def c_series(Cs):
    ovc = 0.
    for c in Cs:
        ovc += (1/c)
    return 1/ovc


def c_series_nc(n,c):
    return c/n
