
import numpy as np
from scipy.linalg import norm

from  . system_of_units import *

from . pynext_types import Cylinder
from . pynext_types import Ray
from . pynext_types import FiberWLS
from . pynext_types import Verbosity
from . pynext_types import vprint, vpblock

from typing import Tuple, List


def throw_dice(dice : float)->bool:
    """Throws a random number and compares with value of dice. Returns true if random < dice"""
    cond = False
    if np.random.random() < dice:
        cond = True
    return cond


def sample_spherical(npoints: int, ndim: int=3)->np.array:
    """Generate points distributed in the surface of a unit sphere.
    The points are in a matrix ((x1,x2,x3...xn), (y1,y2,y3...yn), (z1,z2,z3...zn))
    where n is the number of random points

    """
    vec =  np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def vectors_spherical(npoints: int, ndim: int=3)->np.array:
    """Generate vectors distributed in the surface of a unit sphere.
    The vectros are in a matrix ((x1,y1,z1), (x2,y2,z2)... (xn,yn, zn))
    where n is the number of random points

    """
    return sample_spherical(npoints).T


def generate_random_point_around_p(c: Cylinder, p: np.array, scale: float=1)->np.array:
    """Generates a random point in a sphere around p"""

    vs = scale * vectors_spherical(npoints=1)
    r = vs[0] + p
    return r


def random_point_around_p_inside_cylinder(c: Cylinder, p : np.array, verbose: bool =False,
                                          scale: float =1, maxt: int=1000)->np.array:
    """Generates a random point in a sphere around p. Discards points outside cylynder"""

    cnd = True
    cnt = 0
    scale = 1
    while cnd and cnt < maxt:
        cnt+=1
        r = generate_random_point_around_p(c, p, scale)
        if verbose:
            print(f'generating random point {r}')
        pic = point_inside_cylinder(c, r)
        if pic:
            if verbose:
                print(f'point inside cylinder')
            cnd = False
        else:
            if verbose:
                print(f'point outside cylinder, shooting again')
    return r


def in_endcaps(c: Cylinder, p : np.array)->bool:
    """Returns True if point in end-caps of cyinder"""
    close = np.isclose(np.array([p[2],p[2]]), np.array([c.zmin, c.zmax]), atol=1e-06)
    return close.any()


def cylinder_intersection_roots(r: Ray, c: Cylinder, eps: float =1e-9)->np.array:
    """Computes intersection roots between a ray and a cylinder"""

    a = r.d[0]**2 + r.d[1]**2
    b = 2 * (r.e[0] * r.d[0] + r.e[1] * r.d[1])
    c = r.e[0]**2 + r.e[1]**2 - c.r**2

    roots =  np.roots([a,b,c])
    proots = [x for x in roots if x>eps ]
    if proots:
        return np.min(proots)
    else:
        return 0


def ray_intersection_with_cylinder_end_caps(r: Ray, c: Cylinder, t: float)->np.array:
    """Intersection between a ray and the end-cups of a cylinder"""
    p = r.ray(t)
    if p[2] > c.zmax:
        t = (c.zmax - r.e[2])/r.d[2]
    else:
        t = (c.zmin - r.e[2])/r.d[2]

    return t, r.ray(t)


def ray_intersection_with_cylinder(r: Ray, c:Cylinder)->Tuple[float,np.array]:
    """Intersection between a ray and a cylinder"""
    t = cylinder_intersection_roots(r, c)
    P = r.ray(t)
    z = P[2]
    if z < c.zmin or z > c.zmax:
        t, P = ray_intersection_with_cylinder_end_caps(r, c, t)
    return t, P


def reflected_ray_in_cylinder(r: Ray, c:Cylinder)->np.array:
    """specular reflection in a cylinder
    Given an incident ray I and a normal N, the reflected ray R is:
    vect(R) = vect(I) - 2 (vect(N) cdot vect(I)) vect(I)

    """
    _, P =  ray_intersection_with_cylinder(r, c)
    N    =  c.normal_to_barrel(P)
    I    =  r.d
    NI   =  np.dot(N, I)
    return I -2 * NI * I


def reflected_ray(I:np.array, N:np.array)->np.array:
    """specular reflection in a cylinder
    Given an incident ray I and a normal N, the reflected ray R is:
    vect(R) = vect(I) - 2 (vect(N) cdot vect(I)) vect(I)

    """
    NI   =  np.dot(N, I)
    return I -2 * NI * N


def vuv_transport(c : Cylinder, p=np.array([0,0,0]), nphotons: int=10)->np.array:
    """VUV transport is a short name for
    generate_photons_in_point_p_inside_cylinder_and_propagate_to_cylinder_surface()
    The idea is that the photons generated in a point inside the cylinder are VUV
    photons created by S1 or S2 signals. Those photons propagate to the detector
    light tubes

    """

    n = int(nphotons)
    R = vectors_spherical(n) #
    VUV = np.zeros((n,3))

    for i in np.arange(n):
        d = R[i]
        r = Ray(p,d)
        t, P = ray_intersection_with_cylinder(r, c)
        VUV[i,0:3] = P
        #VUV[i,3] = t
    return VUV


def vuv_to_blue_transport_from_point(c: Cylinder, p : np.array,
                                     scale: float=1, verbose: bool =False)->np.array:
    """A VUV photon impinging the cylinder c on point p, is WLS by TPB
    and re-emitted isotropically, then transported to the cylinder walls

    """

    rp   = generate_random_point_around_p(c, p, scale)
    d    = unit_vectors_from_two_points(p,rp)
    r    = Ray(p,d)  # ray from point p in the direction of d
    t, P = ray_intersection_with_cylinder(r, c)

    if verbose:
        print(f'Reflecting from point {p} to point {rp}')
        print(f'Intersection root {t} intersection point {P}')
        print(f'Point in cylynder? {point_in_cylinder(c, P)}')

    return P


def blue_to_blue_transport_from_point(c: Cylinder, p : np.array,
                                      scale: float=1, verbose: bool=False)->np.array:
    """A VUV photon impinging the cylinder c on point p, is WLS by TPB
    and re-emitted isotropically, then transported to the cylinder walls

    """


    rp   = random_point_around_p_inside_cylinder(c, p, verbose)
    d    = unit_vectors_from_two_points(p,rp)
    r    = Ray(p,d)  # ray from point p in the direction of d
    t, P = ray_intersection_with_cylinder(r, c)

    if verbose:
        print(f'Reflecting from point {p} to point {rp}')
        print(f'Intersection root {t} intersection point {P}')
        print(f'Point in cylynder? {point_in_cylinder(c, P)}')

    return P


def vuv_to_blue_transport(c: Cylinder, VUV : np.array,
                          scale: float =1, verbose: bool=False)->np.array:

    """ Photons in the VUV array are shifted and propagated one to one"""

    BLUE = np.zeros((VUV.shape[0], 3))
    for i, p in enumerate(VUV):
        if verbose:
            print(f'shifting vuv photon from point {p}')
        P = vuv_to_blue_transport_from_point(c, p, scale, verbose)
        if verbose:
            print(f'shifted photons intersects cylinder at {P}')
        BLUE[i,0:3] = P
    return BLUE



def blue_to_green_transport(c: Cylinder, BLUE : np.array,  fwls: FiberWLS, gridTrans: float = 0.9,
                            nphotons: int = 1e+9,scale : int =1,
                            verbose : bool =False)->Tuple[int, np.array]:
    """ Photons in the BLUE array are propagated and converted into green photons"""
    NR = []
    ngreen=0
    nabsPTFE = 0
    nabsFiber = 0
    nendFiber = 0

    if nphotons > BLUE.shape[0]:
        nphotons = BLUE.shape[0]

    for i in range(nphotons):
        p    = BLUE[i]
        nr   = 0
        blue = True
        if verbose:
            print(f'reflecting blue photon number {i} from point {p}')

        while blue:
            if verbose:
                print(f'photon number {i} reflection number {nr}')

            if in_endcaps(c, p):  #photon in endcaps. Increase reflection counter and adjust weight
                if verbose:
                    print(f'photon is  in endcaps: reflect it')

                if throw_dice(dice = fwls.qptfe * gridTrans):
                    nr += 1
                    if verbose:
                        print(f'reflect photon, nr = {nr}')
                    p = blue_to_blue_transport_from_point(c, p, verbose)
                else:
                    if verbose:
                        print(f' photon absorbed by PTFE')
                    nabsPTFE +=1
                    blue = False

            else: #photon in barrel
                if verbose:
                    print(f'photon is in barrel')
                if throw_dice(dice = fwls.blue_absorption_probability):   # photon absorbed by the fiber
                    nabsFiber += 1
                    if verbose:
                        print(f'photon absorbed by fiber')

                    if throw_dice(dice = fwls.quantum_efficiency):   # green photon re-emited
                        if verbose:
                            print(f'green photon re-emitted')

                        # if throw_dice(dice = fwls.trapping_efficiency()):   # green photon trapped
                        #     if verbose:
                        #         print(f'green photon trapped by fiber')
                        NR.append(nr)
                        ngreen += 1
                        blue = False
                        # else:
                        #     if verbose:
                        #         print(f'green photon not trapped by fiber')
                        #     blue = False
                    else:
                        nendFiber += 1
                        if verbose:
                            print(f'green photon not re-emitted')
                        blue = False
                else:
                    if verbose:
                        print(f'photon not absorbed by fiber')
                    if throw_dice(dice = fwls.qptfe):
                        nr += 1
                        if verbose:
                            print(f'reflect photon, nr = {nr}')
                        p = blue_to_blue_transport_from_point(c, p,verbose)
                    else:
                        if verbose:
                            print(f' photon absorbed by PTFE')
                        nabsPTFE +=1
                        blue = False

    return ngreen, nabsPTFE, nabsFiber, nendFiber, np.array(NR)


def count_number_appearance(x : np.array, value: float)->int:
    """Counts the number of times that an element with value appears in array x"""

    return list(x.flatten()).count(value)


def xyz_from_vuv(vuv):
    """takes vector vuv ((x1,y1,z1,t1)...(xn,yn,zn,tn)) and returns
    ((x1,x2...xn), (y1, y2...yn), (z1, z2...zn))
    """

    xi,yi,zi = vuv[:,0:3].T
    return xi,yi,zi


def xyz_from_points(P):
    xi,yi,zi = P[:].T
    return xi,yi,zi


def vuv_fractions(vuv : np.array, zmin : float, zmax : float)->Tuple[float, float, float]:
    _,_,zi = xyz_from_vuv(vuv)
    fzmin = count_number_appearance(zi, zmin)
    fzmax = count_number_appearance(zi, zmax)
    ntot   = zi.shape[0]
    return fzmin/ntot, fzmax/ntot, (ntot-fzmin-fzmax)/ntot


def point_inside_cylinder(c: Cylinder, p : np.array):
        s1 = np.sqrt(p[0]**2 + p[1]**2) <= c.r
        s2 = c.zmin <= p[2] <= c.zmax
        return s1 and s2


def point_in_cylinder(c: Cylinder, p : np.array, eps=1e-7):
    r = np.sqrt(p[0]**2 + p[1]**2)
    z = p[2]
    print(f'r ={r}, z = {z}')
    s1 = np.abs(r - c.r) <= eps
    s2 = np.abs(z - c.zmin) <= eps
    s3 = np.abs(z - c.zmax) <= eps
    return s1 or s2 or s3


def unit_vectors_from_two_points(p0 : np.array, p1 :np.array)->np.array:
    v = p1 - p0
    mag = norm(v)
    return v / mag
