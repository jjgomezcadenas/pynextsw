from dataclasses import dataclass, field
import abc
import numpy as np
import pandas as pd
from scipy.linalg import norm
from scipy.special import erfc
from  . system_of_units import *
from  . CylinderGeomEff import barrel_detection_efficiency

from typing      import Tuple
from typing      import Dict
from typing      import List
from typing      import TypeVar
from typing      import Optional
from enum        import Enum

Number = TypeVar('Number', None, int, float)
Str   = TypeVar('Str', None, str)
Range = TypeVar('Range', None, Tuple[float, float])
Array = TypeVar('Array', List, np.array)
Int = TypeVar('Int', None, int)

### Hamamatsu data on DC
tC = np.array([-10, 0,  10,  20,  30,   40,   50])  # Temparature in C
dR = np.array([ 10,30,  90, 250, 750, 2000, 6000]) # in kHz
ldR = np.log(dR)
coeffs = np.polyfit(tC, ldR, deg=1)
mx, bx = coeffs
y_fit = np.polyval(coeffs, tC)  # fitted function 

def fdcvst(m, b):
    def dvcst(t):
        return b + m * t
    return dvcst

logdc_vs_t = fdcvst(mx, bx) 

def dc_mm2(t):
    """
    Get the DC rate per mm2, using Hamamatsu data
    the S13360 series has size 3 x 3 mm2, thus we divide by 9 to get DC
    
    """
    ldc = logdc_vs_t(t)
    dc = np.exp(ldc)/9
    return dc * kilohertz/mm2



rhoXe = {'xe_2020': (20*bar, 124.3 * kg/m3),
         'xe_3020': (20*bar, 203.35 * kg/m3),
         'xe_1520': (15*bar, 89.9 * kg/m3),
         'xe_1020': (10*bar,   58 * kg/m3),
         'xe_0520': ( 5*bar,   30 * kg/m3),
         'xe_0720': ( 7*bar,   40 * kg/m3),
         'xe_0920': ( 9*bar,   50 * kg/m3),
         'xe_10m50':(10*bar,   80 * kg/m3),
         'xe_15m40':(15*bar,  126 * kg/m3),
         'lxe'    : (1*bar,   3  * g/cm3)
         }
 
class Verbosity(Enum):
    mute     = 0
    concise  = 1
    chat     = 2
    verbose  = 3
    vverbose = 4


def vprint(msg, verbosity, level=Verbosity.mute):
    if verbosity.value <= level.value and level.value >0:
        print(msg)
        

def vpblock(msgs, verbosity, level=Verbosity.mute):
    for msg in msgs:
        vprint(msg, verbosity, level)



@dataclass
class Shape(abc.ABC):
    @property
    def area(self)->float:
        pass
    @property
    def volume(self)->float:
        pass


@dataclass
class Cylinder(Shape):
    r   : float
    zmin: float
    zmax: float
    #p0  : np.array = field(init=False)
    #p1  : np.array = field(init=False)

    def __post_init__(self):
        self.p0 = np.array([0, 0, self.zmin]) # point in one endcup
        self.p1 = np.array([0, 0, self.zmax]) # point in the other

        mag, v, n1, n2 = self.unit_vectors_()
        self.P, self.P2, self.P3 = self.surfaces_(mag,  v, n1, n2)

    def normal_to_barrel(self, P : np.array)->np.array:
        """Normal to the cylinder barrel
        Uses equation of cylynder: F(x,y,z) = x^2 + y^2 - r^2 = 0
        then n = Grad(F)_P /Norm(Grad(F))_P
        n = (2x, 2y, 0)/sqrt(4x^2 + 4y^2) = (x,y,0)/r (P)
        """
        return np.array([P[0], P[1], 0]) / self.r

    def cylinder_equation(self, P : np.array)->float:
        return P[0]**2 + P[1]** 2 - self.r**2

    @property
    def length(self)->float:
        return self.zmax - self.zmin
    
    @property
    def radius(self)->float:
        return self.r

    @property
    def perimeter(self)->float:
        return 2 * np.pi * self.r

    @property
    def area_barrel(self)->float:
        return 2 * np.pi * self.r * self.length

    @property
    def area_endcap(self)->float:
        return np.pi * self.r **2

    @property
    def area(self)->float:
        return self.area_barrel + 2 * self.area_endcap

    @property
    def volume(self)->float:
        return np.pi * self.r **2 * self.length

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

    def __str__(self):
        s= f"""
        Cylinder: radius ={self.radius/mm} mm, length = {self.length/mm} mm
        """
        return s

    __repr__ = __str__

@dataclass
class Sphere(Shape):
    r   : float

    def __post_init__(self):
        self.phi = np.linspace(0, np.pi, 20)
        self.theta = np.linspace(0, 2 * np.pi, 40)
        self.x = self.r * np.outer(np.sin(self.theta), np.cos(self.phi))
        self.y = self.r * np.outer(np.sin(self.theta), np.sin(self.phi))
        self.z = self.r * np.outer(np.cos(self.theta), np.ones_like(self.phi))

    @property
    def area(self)->float:
        return 4 * np.pi * self.r**2

    @property
    def volume(self)->float:
        return (4/3) *np.pi * self.r **3


@dataclass
class Ray:
    e   : np.array
    d   : np.array
    def ray(self,t):
        return self.e + t * self.d
    @property
    def unit_vector(self):
        v = self.d - self.e
        return v / norm(v)


@dataclass
class WLS:
    name : str
    qeff : float

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


    @property
    def quantum_efficiency(self):
        return self.qeff

    def emission_tpb(self, lamda):
        return self.emission_tpb_(lamda/nm) / self.emx


@dataclass
class FiberWLS:
    d      : float  #diameter for cyl, lateral size for square
    wls    : WLS
    qfib   : float  # quantum efficiency
    qptfe  : float  # reflectivity of PTFE around fiber
    ncore  : float
    nclad1 : float
    nclad2 : float
    latt   : float  # attenuation length  d
    path   : str = '/Users/jjgomezcadenas/Projects/toyXeDetectors/pynextsw/pynext/'

    def __post_init__(self):
        # TIR in core to clad 1
        self.thetac1 = np.arcsin(self.nclad1/self.ncore)
        self.thetat1  = 0.5 * np.pi - self.thetac1
        self.ptir1 = (1 - np.cos(self.thetat1))  # 2 x 2 pi (1 - cos(theta)) /4 pi (forward and backward)
        #self.ptir1 = np.cos(self.thetac1)  # 2 x 2 pi (cos(theta) /4 pi (forward and backward)

        # refracted to clad2 : critical
        self.theta2 = np.arcsin((self.nclad1 / self.nclad2) * np.sin(self.thetac1))

        # TIR in clad1 to clad2
        self.thetac2 = np.arcsin(self.nclad2/self.nclad1)
        self.thetat2  = 0.5 * np.pi - self.thetac2
        self.ptir2 = (1 - np.cos(self.thetat2))  # fraction between two claddings
        #self.ptir2 = np.cos(self.thetac2)

        self.teff1 = self.qfib * self.ptir1
        self.teff2 = self.qfib * self.ptir2
        self.fabs = pd.read_csv(self.path+'stGobainAbs.csv', delimiter=',')
        self.fem = pd.read_csv(self.path+'stGobainEm.csv', delimiter=',')
        self.fmu = pd.read_csv(self.path+'elijanMu.csv', delimiter=',')
        L        = np.arange(350, 600, 1) * nm
        num      = np.sum(self.wls.emission_tpb(L) *  self.blue_absorption(L) )
        den      = np.sum(self.wls.emission_tpb(L))
        self.Pa  = num / den

    @property
    def quantum_efficiency(self)->float:
        return self.qfib

    @property
    def diameter(self)->float:
        return self.d

    @property
    def trapping_efficiency_c1(self)->float:
        return self.teff1

    @property
    def trapping_efficiency_c2(self)->float:
        return self.teff2

    @property
    def trapping_efficiency(self)->float:
        return self.teff1
        #+ self.teff2  Just consider trapping efficiency inside fiber

    def transmittance(self, d)->float:
        return np.exp(-d / self.latt)

    def absorption(self, d)->float:
        return 1 - self.transmittance(d)

    def blue_transmittance(self, lamda)->float:
        mu = self.wls_mu(lamda)
        dcm = self.d/cm
        #print(lamda/nm, dcm, mu, np.exp(-dcm * mu))
        return np.exp(-dcm * mu)

    def blue_absorption(self, lamda)->float:
        return 1 - self.blue_transmittance(lamda)

    @property
    def blue_absorption_probability(self)->float:
        return  self.Pa

    def wls_absorption(self, lamda : float)->float:
        return np.interp(lamda/nm, self.fabs.WL.values, self.fabs.A.values)

    def wls_emission(self, lamda : float)->float:
        return np.interp(lamda/nm, self.fem.WL.values, self.fem.A.values)

    def wls_mu(self, lamda : float)->float:
        return np.interp(lamda/nm, self.fmu.WL.values, self.fmu.mu.values)

    def __str__(self):
        s= f"""
        diameter ={self.d/mm} mm, Q = {self.qfib}, PTFE refl = {self.qptfe}
        ncore = {self.ncore}, nclad1 ={self.nclad1}, nclad2 ={self.nclad2}
        Absoprtion prob at 450 nm     = {self.blue_absorption(450*nm)}
        Trapping efficieny            = {self.trapping_efficiency}
        Fiber coated with WLS         = {self.wls.name}
        WLS QE                        = {self.wls.quantum_efficiency}

    """
        return s

    __repr__ = __str__



@dataclass
class SiPM:
    name   : str
    xsize  : float
    PDE    : float
    #Rs     : float # Series resistance connection
    #C      : float # capacitance
    #path   : str = '/Users/jjgomezcadenas/Projects/Development/pynextsw/pynext/'

    
    def __post_init__(self):

        def fdcvst(m, b):
            def dvcst(t):
                return b + m * t
            return dvcst

        tC = np.array([-10, 0,  10,  20,  30,   40,   50])  # Temparature in C
        dR = np.array([ 10,30,  90, 250, 750, 2000, 6000]) # in kHz
        ldR = np.log(dR)
        coeffs = np.polyfit(tC, ldR, deg=1)
        mx, bx = coeffs
        self.logdc_vs_t = fdcvst(mx, bx)
    

        #self.fpde = pd.read_csv(self.path+'s13360.csv', delimiter=',')
        #tf        = np.polyfit(self.fdcr.K.values, np.log(self.fdcr.DCR.values), 2)
        #self.fdcr = pd.read_csv(self.path+'dcrt2.csv', delimiter=',')
        #self.TK   = np.poly1d(tf)
        #self.F = 1.0 # to achieve the calibrated value o DCR

    @property
    def area(self):
        return self.xsize**2
    
    @property
    def pde(self):
        return self.PDE

    # def pde(self, lamda : float)->float:
    #     return np.interp(lamda, self.fpde.WL.values, self.fpde.PDE.values/100)

    #def log_dcr(self, t : float)->float: # t in 1/K, K kelvin
    #    return self.TK(t)

    def dcr_sipm_per_unit_area (self, tC : float)->float:  # tC in Celsius
        #tK = tC + 273.15
        #ldcr = self.log_dcr(1/tK)
        #return self.F* np.exp(ldcr) * hertz / mm2
        ldc = self.logdc_vs_t(tC)   
        return np.exp(ldc)/9 * kilohertz / mm2 # this corresponds to the area of the SiPMs 9 mm2

    def dcr_sipm_per_time_and_area (self, tC : float, time: float) ->float:
        return  time * self.dcr_sipm_per_unit_area(tC)

    def __str__(self):
        u = hertz / mm2
        s =f"""
        sensor ={self.name}, size = {self.xsize/mm} mm, PDE = {self.PDE}
        DCR per u.a. (tC = 20C) = {self.dcr_sipm_per_unit_area (20)/u:.2g} Hz/mm2;
        DCR per u.a time window (tC = 20C, t=500 ns) = {self.dcr_sipm_per_time_and_area(20, 500*ns)/mm2:.2g} cs/mm2;
        """

        return s

    __repr__ = __str__


@dataclass
class DTP:
    sipm   : SiPM
    r      : float # radius of DTP
    ff     : float # fill factor
    pitch  : float = 10 * mm

    @property
    def area(self):
        return np.pi * self.r**2
    
    @property
    def n_sipm(self):
        return self.area / self.pitch**2
    
    def dcr_sipm(self, tC : float):
        return self.ff * self.area * self.sipm.dcr_sipm_per_unit_area (tC)
    
    def dcr_sipm_per_time(self, tC : float, time: float) ->float:
        return  time * self.dcr_sipm(tC)

    def __str__(self):
        u = hertz / mm2
        s =f"""
        DTP: Fill factor = {self.ff}
        area = {self.area/mm2:.2g} mm2, number of SiPMs = {self.n_sipm:.3g}
        SiPMs: {self.sipm}
        DCR  (tC = 20C) = {self.dcr_sipm(20)/u:.2g} Hz;
        DCR  time window (tC = 20C, t=500 ns) = {self.dcr_sipm_per_time(20, 500*ns):.2g} counts;
        
        """
        #DCR  (tC = -40C) = {self.dcr_sipm(-40)/u:.2g} Hz;
        #DCR  time window (tC = -40C, t=500 ns) = {self.dcr_sipm_per_time(-40, 500*ns)/mm2:.2g} counts;

        return s

    __repr__ = __str__


@dataclass
class TpcEL:
    """
    Defines a EL TPC
    EP = E/P
    dV = drift voltage
    P  = pressure
    d  = EL grid gap
    L  = drift lenght
    Ws = energy to produce a scintillation photon
    Wi = energy to produce a ionization photon
    """
    EP : float = 3.5 * kilovolt / (cm * bar)
    dV : float = 0.5 * kilovolt / cm
    P  : float =  15 * bar
    d  : float =   5 * mm
    L  : float = 120 * cm
    Ws : float =  39.2 * eV
    Wi : float =  21.9 * eV

    def __post_init__(self):
        u = kilovolt / (cm * bar)
        ep = self.EP/u
        self.Vgrid = self.EP * self.d * self.P
        self.Vc    = self.Vgrid + self.dV * self.L
        self.YP    = 140 * ep - 116  # in photons per electron bar^-1 cm^-1
        self.Ng    = self.YP * self.d/cm * self.P/bar #photons/e

    @property
    def e_over_p(self):
        return self.EP

    @property
    def drift_voltage(self):
        return self.dV

    @property
    def grid_voltage(self):
        return self.Vgrid

    @property
    def cathode_voltage(self):
        return self.Vc

    @property
    def pressure(self):
        return self.P

    @property
    def grid_distance(self):
        return self.d

    @property
    def drift_length(self):
        return self.L

    @property
    def optical_gain(self):
        return self.Ng

    def scintillation_photons(self, E):
        return E / self.Ws

    def el_photons(self, E):
        return self.ionization_electrons(E) * self.optical_gain

    def ionization_electrons(self, E):
        return E / self.Wi


    def __str__(self):
        kv_cm_b = kilovolt / (cm * bar)
        kv_cm   = kilovolt / cm
        cm_bar = 1 / (cm * bar)
        s= """
        E/P = %7.2f kV * cm^-1* bar^-1
        dV = drift voltage = %7.2f kV * cm^-1
        P  = pressure = %7.2f bar
        d  = EL grid gap = %7.2f mm
        L  = drift lenght =%7.2f m
        Grid voltage = %7.2f kV
        Cathode voltage = %7.2f kV
        Yield =  %7.2e photons/e

    """%(self.e_over_p / kv_cm_b,
         self.drift_voltage / kv_cm,
         self.pressure / bar,
         self.grid_distance / mm,
         self.drift_length / m,
         self.grid_voltage / kilovolt,
         self.cathode_voltage /kilovolt,
         self.optical_gain)

        s+="""
        Primary scintillation photons per MeV = %7.2e
        Primary ionization electrons per MeV = %7.2e
        EL photons per MeV                   = %7.2e
        """%(self.scintillation_photons(1 * MeV),
             self.ionization_electrons(1 * MeV),
             self.el_photons(1 * MeV))
        s+="""
        Primary scintillation Krypton = %7.2e
        Primary ionization electrons Krypton = %7.2e
        EL photons Krypton                   = %7.2e
        """%(self.scintillation_photons(41.5 * keV),
             self.ionization_electrons(41.5 * keV),
             self.el_photons(41.5 * keV))
        s+="""
        Primary scintillation Qbb = %7.2e
        Primary ionization electrons Qbb = %7.2e
        EL photons Qbb                   = %7.2e
        """%(self.scintillation_photons(2458 * keV),
             self.ionization_electrons(2458 * keV),
             self.el_photons(2458 * keV))

        return s

    __repr__ = __str__

@dataclass
class TpcXe:
    """A cylynder filled with Xenon"""
    cyl  : Cylinder
    name : str = 'xe_1520'
    p    : float = 15 * bar
    rho  : float = 89.9 * kg/m3


    def __post_init__(self):

       
        self.p = 0
        self.rho = 0
        for key, value in rhoXe.items():
            if key == self.name:
                self.p = value[0]
                self.rho = value[1]
        
        if self.p == 0:
            print(f"Configuration requested = {self.name} not found")
            print(f"Taking default (15 bar, 20C)")
            self.p = 20 * bar
            self.rho = 89.9 *  kg/m3
        
        #self.ps   = (10,15,20,30, 40)
        #self.rhos = (58, 89.9, 124.3, 203.35, (203.35) *(4/3))

    @property
    def pressure(self):
        return self.p

    @property
    def density(self):
        return self.rho 
        #P = self.p/bar
        #return np.interp(P, self.ps, self.rhos) * kg / m3

    @property
    def mass(self):
        return self.cyl.volume * self.density

    def __str__(self):
        g_cm3 = g / cm3
        cm2_g = cm2 / g
        icm   = 1 / cm

        s= f"""
        Configuration  = {self.name:s}
        density (rho)  = {self.density/g_cm3:7.2f} g/cm3
        mass           = {self.mass/kg:7.2f}  kg
        Cylinder       = {self.cyl}
    """

        return s

    __repr__ = __str__


@dataclass
class FiberDetector:
    tpcel     : TpcEL
    tpcxe     : TpcXe
    fwls      : FiberWLS  # fiber type
    sipm      : SiPM
    eff_t     : float # transport efficiency of the detector.
    sampling  : float
    #adcPerPes : float
    tempC      : float

    def __post_init__(self):
        self.transport = self.eff_t * self.fwls.wls.quantum_efficiency * self.fwls.trapping_efficiency
        self.attenuation = self.fwls.transmittance(self.tpcxe.cyl.length/2)


    @property
    def operating_temperatureC (self):
        return self.tempC

    @property
    def n_fibers (self):
        return int(self.tpcxe.cyl.perimeter / self.fwls.diameter) + 1

    @property
    def fiber_efficiency (self): 
        return self.transport * self.attenuation
    
    @property
    def total_efficiency (self):
        return self.fiber_efficiency * self.sipm.PDE
    
    def dcr_sipm(self, tC : float): # area 1 mm2
        return self.n_fibers  * self.sipm.dcr_sipm_per_unit_area (tC)
    
    def dcr_sipm_per_time(self, tC : float, time: float) ->float:
        return  time * self.dcr_sipm(tC)
    

    def __str__(self):
        ren = self.total_efficiency/ self.n_fibers
        
        s =f"""
        gas pressure  = {self.tpcxe.pressure/bar:7.2f} bar
        gas density   = {self.tpcxe.density/(g/cm3):7.2f} g/cm3
        gas mass      = {self.tpcxe.mass/kg:7.2f}  kg

        Dimensions(Cylinder)  = {self.tpcxe.cyl}

        Fibers efficiency = {100*self.fiber_efficiency:.2f} %
        of which: Transport = {100 * self.transport:.2f} % & attenuation = {100 * self.attenuation:.2f} %
        Total detection efficiency = {100*self.total_efficiency:.2f} %
        SiPM PDE          = {self.sipm.PDE:.2f}
        SiPM size         = {self.sipm.xsize/mm:.2f} mm
        fiber size        = {self.fwls.diameter/mm:.2f} mm
        Sampling S1       = {self.sampling/ns:.2f} ns
        number of fibers  = {self.n_fibers:d}

        Primary scintillation Krypton          = {self.tpcel.scintillation_photons(41.5 * keV):.2e}
        EL photons Krypton                     = {self.tpcel.el_photons(41.5 * keV):.2e}
        Primary scintillation Qbb              = {self.tpcel.scintillation_photons(2458 * keV):.2e}
        EL photons Qbb                         = {self.tpcel.el_photons(2458 * keV):.2e}

        Primary scintillation Krypton detected = {self.tpcel.scintillation_photons(41.5 * keV)* self.total_efficiency:.2e}
        EL photons Krypton detected            = {self.tpcel.el_photons(41.5 * keV)* self.total_efficiency:.2e}
        Primary scintillation Qbb detected     = {self.tpcel.scintillation_photons(2458 * keV)* self.total_efficiency:.2e}
        EL photons Qbb detected                = {self.tpcel.el_photons(2458 * keV)* self.total_efficiency:.2e}

        Primary scintillation Krypton det/fiber = {self.tpcel.scintillation_photons(41.5 * keV) * ren:.2e}
        EL photons Krypton det/fiber            = {self.tpcel.el_photons(41.5 * keV)* ren:.2e}
        Primary scintillation Qbb det/fiber     = {self.tpcel.scintillation_photons(2458 * keV)* ren:.2e}
        EL photons Qbb det/fiber                = {self.tpcel.el_photons(2458 * keV)* ren:.2e}

        Number of DCR photons in the detector for:
         --operating temperature ({self.operating_temperatureC:.2f} C)
         --sampling time of {self.sampling/ns:.2f}
         -- nDCR = {self.dcr_sipm_per_time(self.tempC, self.sampling):.2f}
        """

        return s

    __repr__ = __str__


@dataclass
class DTPDetector:
    tpcel     : TpcEL
    tpcxe     : TpcXe
    dtp       : DTP
    eff_grid  : float # grid transparency.
    sampling  : float
    tempC      : float
    

    def __post_init__(self):

        Lhalf = (self.tpcxe.cyl.zmax - self.tpcxe.cyl.zmin) / 2
        Rg = self.tpcxe.cyl.radius
        self.eff_barrel_s1 = barrel_detection_efficiency(Rg, Lhalf) # L in the equation is semilength 
        self.eff_ecup_s1 = 1 - self.eff_barrel_s1
        
    @property
    def eff_geom_s1 (self):
        return self.eff_ecup_s1
    
    @property
    def eff_geom_s2 (self):
        return 0.5 # half of the S2 light hits the DTP

    @property
    def operating_temperatureC (self):
        return self.tempC

    @property
    def n_sipms (self):
        return int(self.dtp.nsipm) + 1

    @property
    def efficiency_s1 (self):
        return self.eff_geom_s1 * self.eff_grid**2 * self.dtp.sipm.pde
    
    @property
    def efficiency_s2 (self):
        return self.eff_geom_s2 * self.eff_grid * self.dtp.sipm.pde

    def __str__(self):
        pskd = self.tpcel.scintillation_photons(41.5 * keV)* self.efficiency_s1 * self.dtp.ff
        elskd = self.tpcel.el_photons(41.5 * keV)* self.efficiency_s2* self.dtp.ff
        psqd = self.tpcel.scintillation_photons(2458 * keV)* self.efficiency_s1 * self.dtp.ff
        elsqd = self.tpcel.el_photons(2458 * keV)* self.efficiency_s2 * self.dtp.ff
        
        s =f"""
        gas pressure  = {self.tpcxe.pressure/bar:7.2f} bar
        gas density   = {self.tpcxe.density/(g/cm3):7.2f} g/cm3
        gas mass      = {self.tpcxe.mass/kg:7.2f}  kg

        Dimensions(Cylinder)  = {self.tpcxe.cyl}

        Light efficiency (s1) = {100*self.efficiency_s1:.2f} %
        Light efficiency (s2) = {100*self.efficiency_s2:.2f} %
        SiPM PDE          = {self.dtp.sipm.pde:.2f}
        SiPM size         = {self.dtp.sipm.xsize/mm:.2f} mm
        Sampling S1       = {self.sampling/ns:.2f} ns

        Primary scintillation Krypton          = {self.tpcel.scintillation_photons(41.5 * keV):.2e}
        EL photons Krypton                     = {self.tpcel.el_photons(41.5 * keV):.2e}
        Primary scintillation Qbb              = {self.tpcel.scintillation_photons(2458 * keV):.2e}
        EL photons Qbb                         = {self.tpcel.el_photons(2458 * keV):.2e}

        Primary scintillation Krypton detected = {pskd:.2e}
        EL photons Krypton detected            = {elskd:.2e}
        Primary scintillation Qbb detected     = {psqd:.2e}
        EL photons Qbb detected                = {elsqd:.2e}


        Number of DCR photons in the detector for:
         --operating temperature ({self.operating_temperatureC:.2f} C)
         --sampling time of {self.sampling/ns:.2f}
         -- nDCR = {2* self.dtp.dcr_sipm_per_time(self.operating_temperatureC, self.sampling):.2f}
        """

        return s

    __repr__ = __str__
