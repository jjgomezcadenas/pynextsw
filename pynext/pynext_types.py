from dataclasses import dataclass, field
import abc
import numpy as np
import pandas as pd
from scipy.linalg import norm
from scipy.special import erfc
from  . system_of_units import *

@dataclass
class Shape(abc.ABC):
    def area(self)->float:
        pass
    def volume(self)->float:
        pass


@dataclass
class Cylinder(Shape):
    r   : float
    zmin: float
    zmax: float
    p0  : np.array = field(init=False)
    p1  : np.array = field(init=False)

    def __post_init__(self):
        self.p0 = np.array([0, 0, self.zmin]) # point in one endcup
        self.p1 = np.array([0, 0, self.zmax]) # point in the other

        mag, v, n1, n2 = self.unit_vectors_()
        self.P, self.P2, self.P3 = self.surfaces_(mag,  v, n1, n2)

    def length(self)->float:
        return self.zmax - self.zmin

    def area_barrel(self)->float:
        return 2 * np.pi * self.r * self.length()

    def area_endcap(self)->float:
        return np.pi * self.r **2

    def area(self)->float:
        return self.area_barrel() + 2 * self.area_endcap()

    def volume(self)->float:
        return np.pi * self.r **2 * self.length()

    def unit_vectors_(self):
        #vector in direction of axis
        v = self.p1 - self.p0

        #find magnitude of vector
        mag = norm(v)

        #unit vector in direction of axis
        v = v / mag

        # choose (1,0,0) as second axis unless is first axis
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])

        #make vector perpendicular to v and not v
        n1 = np.cross(v, not_v)

        #normalize n1
        n1 /= norm(n1)

        #make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)

        return mag, v,  n1, n2

    def surfaces_(self, mag, v, n1, n2):
        #surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 2)
        theta = np.linspace(0, 2 * np.pi, 100)
        rsample = np.linspace(0, self.r, 2)

        #use meshgrid to make 2d arrays
        t, theta2 = np.meshgrid(t, theta)

        rsample,theta = np.meshgrid(rsample, theta)

        #generate coordinates for surface
        # "Tube"
        X, Y, Z = [self.p0[i] + v[i] * t + self.r * np.sin(theta2) * n1[i] + self.r * np.cos(theta2) *  n2[i] for i in range(3)]
        # "Bottom"
        X2, Y2, Z2 = [self.p0[i] + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in range(3)]
        # "Top"
        X3, Y3, Z3 = [self.p0[i] + v[i]*mag + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in range(3)]
        return (X,Y,Z), (X2, Y2, Z2), (X3, Y3, Z3)


@dataclass
class Sphere(Shape):
    r   : float

    def __post_init__(self):
        self.phi = np.linspace(0, np.pi, 20)
        self.theta = np.linspace(0, 2 * np.pi, 40)
        self.x = self.r * np.outer(np.sin(self.theta), np.cos(self.phi))
        self.y = self.r * np.outer(np.sin(self.theta), np.sin(self.phi))
        self.z = self.r * np.outer(np.cos(self.theta), np.ones_like(self.phi))

    def area(self)->float:
        return 4 * np.pi * self.r**2

    def volume(self)->float:
        return (4/3) *np.pi * self.r **3


@dataclass
class Ray:
    e   : np.array
    d   : np.array
    def ray(self,t):
        return self.e + t * self.d


@dataclass
class TPB:
    q      : float

    def emission_tpb_(self, l):
        A     = 0.782
        alpha = 3.7e-2
        s1    = 15.43
        mu1   = 418.1
        s2    = 9.72
        mu2   = 411.2

        t1 = A * (alpha/2) * np.exp((alpha/2) * (2 * mu1 + alpha * s1**2 - 2 * l))
        t2  = erfc((mu1 + alpha * s1**2 - l)/(s1 * np.sqrt(2)))
        t3 = (1 - A) * (1/np.sqrt(2 * s2**2 * np.pi)) * np.exp(-(l - mu2)**2/(2*s2**2))
        return t1 * t2 + t3

    def __post_init__(self):
        # TIR in core to clad 1
        L = np.arange(350, 550, 1)
        e = self.emission_tpb_(L)
        self.emx = np.max(e)


    def emission_tpb(self, l):
        return self.emission_tpb_(l) / self.emx


@dataclass
class FiberWLS:
    qfib   : float
    ncore  : float
    nclad1 : float
    nclad2 : float
    latt   : float

    def __post_init__(self):
        # TIR in core to clad 1
        self.thetac1 = np.arcsin(self.nclad1/self.ncore)
        self.thetat1  = 0.5 * np.pi - self.thetac1
        self.ptir1 = (1 - np.cos(self.thetat1))  # 2 x 2 pi (1 - cos(theta)) /4 pi (forward and backward)

        # refracted to clad2 : critical
        self.theta2 = np.arcsin((self.nclad1 / self.nclad2) * np.sin(self.thetac1))

        # TIR in clad1 to clad2
        self.thetac2 = np.arcsin(self.nclad2/self.nclad1)
        self.thetat2  = 0.5 * np.pi - self.thetac2
        self.ptir2 = (1 - np.cos(self.thetat2))  # fraction between two claddings

        self.teff1 = self.qfib * self.ptir1
        self.teff2 = self.qfib * self.ptir2
        self.fabs = pd.read_csv('stGobainAbs.csv', delimiter=',')
        self.fem = pd.read_csv('stGobainEm.csv', delimiter=',')

    def trapping_efficiency_c1(self)->float:
        return self.teff1

    def trapping_efficiency_c2(self)->float:
        return self.teff2

    def trapping_efficiency(self)->float:
        return self.teff1 + self.teff2

    def transmittance(self, d)->float:
        return np.exp(-d / self.latt)

    def absorption(self, d)->float:
        return 1 - self.transmittance(d)

    def wls_absorption(self, lamda : float)->float:
        return np.interp(lamda, self.fabs.WL.values, self.fabs.A.values)

    def wls_emission(self, lamda : float)->float:
        return np.interp(lamda, self.fem.WL.values, self.fem.A.values)


@dataclass
class SiPM:
    name   : str
    xsize  : float
    PDE    : float
    C      : float # capacitance
    Rs     : float # Series resistance connection

    def __post_init__(self):


        self.fpde = pd.read_csv('s13360.csv', delimiter=',')
        self.fdcr = pd.read_csv('dcrt2.csv', delimiter=',')
        tf        = np.polyfit(self.fdcr.K.values, np.log(self.fdcr.DCR.values), 2)
        self.TK   = np.poly1d(tf)
        self.F = 1.0 # to achieve the calibrated value o DCR

    def area(self):
        return self.xsize**2

    def pde(self, lamda : float)->float:
        return np.interp(lamda, self.fpde.WL.values, self.fpde.PDE.values/100)

    def log_dcr(self, t : float)->float: # t in 1/K, K kelvin
        return self.TK(t)

    def dcr_sipm_per_unit_area (self, tC : float)->float:  # tC in Celsius
        tK = tC + 273.15
        ldcr = self.log_dcr(1/tK)
        return self.F* np.exp(ldcr) * hertz / mm2

    def dcr_sipm_per_time (self, tC : float, time: float) ->float:
        return (self.area()) * time * self.dcr_sipm_per_unit_area(tC)

    def __repr__(self):
        s =f"""
        sensor ={self.name}, size = {self.xsize/mm} mm, PDE = {self.PDE}
        capacitance = {self.C/pF:.2f} pF; 
        """

        return s
