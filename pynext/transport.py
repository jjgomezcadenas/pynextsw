
import numpy as np
from   scipy.linalg import norm
from   scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import axes3d

from  . system_of_units import *

from . pynext_types import Cylinder
from . pynext_types import Ray
from . pynext_types import FiberWLS
from . pynext_types import Verbosity
from . pynext_types import vprint, vpblock

from typing import Tuple, List

from . graphics    import draw_cylynder_surface

from . pynext_base import vectors_spherical
from . pynext_base import ray_intersection_with_cylinder
from . pynext_base import cylinder_intersection_roots
from . pynext_base import ray_intersection_with_cylinder_end_caps
from . pynext_base import xyz_from_points
from . pynext_base import point_inside_cylinder
from . pynext_base import point_in_cylinder
from . pynext_base import unit_vectors_from_two_points
from . pynext_base import reflected_ray


def fiber_transport(c : Cylinder, p0=np.array([0,0,0]),
                    thetac:         float     = 60 * degree,
                    nphotons:       int       = 1,
                    verbosityLevel: Verbosity = Verbosity.verbose,
                    drawAll:        bool      = False,
                    drawReflected:  bool      = True,
                    nmax:           int       = 1000,
                    eps:            float     = 1e-7,
                    units:          float     = mm,
                    alpha:          float     =0.2,
                    barrelColor:    str       ='blue',
                    cupColor:       str       ='red',
                    figsize:        Tuple[int,int]            =(16,16),
                    DWORLD:         bool                      =False,
                    WDIM:           List[Tuple[float, float]] =((-1,1),(-1,1),(-1,1)))->np.array:
    """light is generated in the center of the cylinder.
    - ray is transported to surface and angle with normal computed.
    - if angle larger than angle of total internal reflection then ray is reflected
    """

    def draw_I(p0, P):
        i = unit_vectors_from_two_points(p0, P)
        dst = euclidean(p0,P)
        tt = np.linspace(0, dst, 100)

        xi = p0[0] + tt * i[0]
        yi = p0[1] + tt * i[1]
        zi = p0[2] + tt * i[2]
        ax.plot(xi, yi, zi)
        xxi,xyi,xzi = xyz_from_points(P)
        ax.scatter(xxi, xyi, xzi, s=25, c='k', zorder=10)

    n = int(nphotons)
    if drawAll or drawReflected:
        draw =True
    else:
        draw = False

    if draw:
        fig = plt.figure(figsize=figsize)
        ax=plt.subplot(111, projection='3d')
        if DWORLD:
            ax.set_xlim3d(WDIM[0][0], WDIM[0][1])
            ax.set_ylim3d(WDIM[1][0], WDIM[1][1])
            ax.set_zlim3d(WDIM[2][0], WDIM[2][1])
        draw_cylynder_surface(c, ax, units, alpha, barrelColor, cupColor)

    NR =[]
    LE =[]
    ntrap = 0
    for i in np.arange(n):
        vprint(f'photon number {i}',
               verbosity=Verbosity.chat, level=verbosityLevel)

        L = 0
        d = vectors_spherical(1)[0]
        r = Ray(p0,d)
        t, P = ray_intersection_with_cylinder(r, c)

        vprint(f' ray reaches point {P}',
               verbosity=Verbosity.verbose, level=verbosityLevel)

        if P[2] > c.zmax or P[2] < c.zmin:
            vprint(f' outside cylinder! P = {P}',
                   verbosity=Verbosity.concise, level=verbosityLevel)
            continue

        if np.abs(c.cylinder_equation(P)) > eps:
            vprint(f' in end cups! P = {P}',
                   verbosity=Verbosity.concise, level=verbosityLevel)
            continue

        if drawAll:
            draw_I(p0, P)

        L+= euclidean(p0,P)
        N = c.normal_to_barrel(P)
        I = r.unit_vector
        theta = np.arccos(np.dot(I,N))

        vpblock((f'Incident ray = {I}, norm(I) ={norm(I)}',
                 f'Normal to surface in P = {N}, norm(N) ={norm(N)}',
                 f'theta (deg) = {theta/degree}'),
                 verbosity=Verbosity.chat, level=verbosityLevel)
        nc = 0
        if theta > thetac:
            ntrap +=1

            while P[2] < c.zmax and P[2] > c.zmin and nc < nmax:
                R = reflected_ray(I, N)

                vpblock((f'reflection number {nc}',
                 f'reflected ray = {R}, norm(R) ={norm(R)}',
                 f'angle I-R = {np.arccos(np.dot(I,R))/degree}'),
                 verbosity=Verbosity.verbose, level=verbosityLevel)

                r = Ray(P,R)
                P0 = P
                t, P = ray_intersection_with_cylinder(r, c)
                N = c.normal_to_barrel(P)
                I = R
                theta = np.arccos(np.dot(I,N))

                if draw:
                    draw_I(P0, P)

                L+= euclidean(P0,P)
                vpblock((f' ray reaches point {P}',
                 f'Incident ray = {I}, norm(I) ={norm(I)}',
                 f'Normal to surface in P = {N}, norm(N) ={norm(N)}',
                 f'theta (deg) = {theta/degree}'),
                 verbosity=Verbosity.chat, level=verbosityLevel)

                nc+=1
            NR.append(nc)
            LE.append(L)
            vprint(f'Photon {i} Reached point P={P}, distance travelled = {L}',
                   verbosity=Verbosity.chat, level=verbosityLevel)
    if draw:
        plt.show()

    return ntrap, NR, LE
