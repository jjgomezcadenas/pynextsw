
import numpy as np
from  . system_of_units import *

from . pynext_types import Cylinder
from . pynext_types import Ray

from typing import Tuple, List

def vector1D(coords : List[float])->np.array:
    return np.array(coord)


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /=    np.linalg.norm(vec, axis=0)
    return vec


def random_point_generator(npoints):
      return np.random.rand(int(npoints), 3)


def intersection_roots(r: Ray, c: Cylinder)->np.array:
    a = r.d[0]**2 + r.d[1]**2
    b = 2 * (r.e[0] * r.d[0] + r.e[1] * r.d[1])
    c = r.e[0]**2 + r.e[1]**2 - c.r**2
    roots =  np.roots([a,b,c])
    return np.min([x for x in roots if x>0 ])


def intersection_end_caps(r: Ray, c: Cylinder, t: float)->np.array:
    p = r.ray(t)
    if p[2] > c.zmax:
        t = (c.zmax - r.e[2])/r.d[2]

    else:
        t = (c.zmin - r.e[2])/r.d[2]

    return t, r.ray(t)


def ray_intersection_with_cylinder(r: Ray, c:Cylinder)->Tuple[float,float]:
        t = intersection_roots(r, c)
        P = r.ray(t)
        z = P[2]
        if z < c.zmin or z > c.zmax:
            t, P = intersection_end_caps(r, c, t)
        return t, P


def vuv_transport(c : Cylinder, p=np.array([0,0,0]), nphotons=10)->np.array:
    """Creates VUV photons at point p and propagates them to cylinder c
    defining the VUV detector"""

    n = int(nphotons)
    R = sample_spherical(n).T #to get the vectors
    VUV = np.zeros((n,4))

    for i in np.arange(n):
        d = R[i]
        r = Ray(p,d)
        t, P = ray_intersection_with_cylinder(r, c)
        VUV[i,0:3] = P
        VUV[i,3] = t
    return VUV

def count_number_appearance(x : np.array, value: float)->int:
    return list(x.flatten()).count(value)


def xyz_from_vuv(vuv):
    xi,yi,zi = vuv[:,0:3].T
    return xi,yi,zi


def vuv_fractions(vuv : np.array, zmin : float, zmax : float)->Tuple[float, float, float]:
    _,_,zi = xyz_from_vuv(vuv)
    fzmin = count_number_appearance(zi, zmin)
    fzmax = count_number_appearance(zi, zmax)
    ntot   = zi.shape[0]
    return fzmin/ntot, fzmax/ntot, (ntot-fzmin-fzmax)/ntot
