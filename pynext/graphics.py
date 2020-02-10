import numpy as np
import pandas as pd
import os, sys

from scipy.linalg import norm

from  . system_of_units import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from . pynext_types import  Cylinder, Sphere
from . pynext_base  import  sample_spherical
from . pynext_base  import  vectors_spherical
from . pynext_base  import  xyz_from_vuv
from . pynext_base  import  xyz_from_points
from . pynext_base  import  point_in_cylinder
from . pynext_base  import  point_inside_cylinder
from . pynext_base  import  unit_vectors_from_two_points
from . pynext_base  import  ray_intersection_with_cylinder
from . pynext_types import Ray

from typing import Tuple, List


def set_fonts(ax, fontsize=20):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


def draw_cylynder_surface(c: Cylinder, ax, units, alpha=0.2, barrelColor='blue', cupColor='red'):
    ax.plot_surface(c.P[0]/units, c.P[1]/units, c.P[2]/units, color=barrelColor, alpha=alpha)
    ax.plot_surface(c.P2[0]/units, c.P2[1]/units, c.P2[2]/units, color=cupColor, alpha=alpha)
    ax.plot_surface(c.P3[0]/units, c.P3[1]/units, c.P3[2]/units, color=cupColor, alpha=alpha)


def draw_cylinder(c : Cylinder, units=mm, alpha=0.2, barrelColor='blue', cupColor='red',
                  figsize=(16,16), DWORLD=False, WDIM=((-1,1),(-1,1),(-1,1))):


    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111, projection='3d')
    if DWORLD:
        ax.set_xlim3d(WDIM[0][0], WDIM[0][1])
        ax.set_ylim3d(WDIM[1][0], WDIM[1][1])
        ax.set_zlim3d(WDIM[2][0], WDIM[2][1])
    draw_cylynder_surface(c, ax, units, alpha,  barrelColor, cupColor)
    #ax.plot_surface(c.P[0], c.P[1], c.P[2], color=barrelColor, alpha=alpha)
    #ax.plot_surface(c.P2[0], c.P2[1], c.P2[2], color=cupColor, alpha=alpha)
    #ax.plot_surface(c.P3[0], c.P3[1], c.P3[2], color=cupColor, alpha=alpha)
    plt.show()


def draw_cylnder_nomal_at_P(P: np.array, c : Cylinder, tscale=1,
                            units=mm, alpha=0.2, barrelColor='blue', cupColor='red',
                            figsize=(16,16)):

    N = c.normal_to_barrel(P)

    def draw_normal(P,N):
        tt = np.linspace(0, tscale, 100)

        xi = P[0] + tt * N[0]
        yi = P[1] + tt * N[1]
        zi = P[2] + tt * N[2]
        ax.plot(xi, yi, zi)

    xi,yi,zi = xyz_from_points(np.array([P]))

    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111, projection='3d')
    draw_cylynder_surface(c, ax, units, alpha, barrelColor, cupColor)

    draw_normal(P, N)
    ax.scatter(xi, yi, zi, s=25, c='k', zorder=10)
    plt.show()


def draw_sphere(s : Sphere,  color='k', rstride=1, cstride=1, figsize=(16,16)):

    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111, projection='3d', aspect='equal')
    ax.plot_wireframe(s.x, s.y, s.z, color=color, rstride=rstride, cstride=cstride)
    plt.show()


def draw_spherical_sample(s : Sphere,  npoints:int, rstride=1, cstride=1, figsize=(16,16)):

    xi, yi, zi = sample_spherical(npoints)

    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111, projection='3d', aspect='equal')
    ax.plot_wireframe(s.x, s.y, s.z, color='k', rstride=rstride, cstride=cstride)
    ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)

    plt.show()


def draw_points_sphere(s : Sphere,  points : Tuple[np.array, np.array, np.array],
                       rstride=1, cstride=1, figsize=(16,16)):

    fig = plt.figure(figsize=figsize)
    xi, yi, zi = points
    ax=plt.subplot(111, projection='3d', aspect='equal')
    ax.plot_wireframe(s.x, s.y, s.z, color='k', rstride=rstride, cstride=cstride)
    ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)

    plt.show()


def draw_random_point_around_point(p: np.array, figsize=(10,10))->np.array:
    vs = vectors_spherical(npoints=1)
    r = vs[0] + p
    P = np.array([p, r])
    sp = Sphere(1)
    draw_points_sphere(sp,  xyz_from_points(P), figsize=figsize)



def draw_vuv(vuv: np.array, c : Cylinder, p=np.array([0,0,0]), tscale=1,
             units=mm, alpha=0.2, barrelColor='blue', cupColor='red', drawRays=True,
             figsize=(16,16)):

    def draw_rays(e, R):
        tt = np.linspace(0, tscale, 100)
        for r in R:
            #t= tscale*r[2]
            #tt = np.linspace(0, t, 100)
            xi = e[0] + tt * r[0]
            yi = e[1] + tt * r[1]
            zi = e[2] + tt * r[2]
            ax.plot(xi, yi, zi)

    xi,yi,zi = xyz_from_points(vuv)

    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111, projection='3d')
    draw_cylynder_surface(c, ax, units, alpha, barrelColor, cupColor)
    #ax.plot_surface(c.P[0], c.P[1], c.P[2], color=barrelColor, alpha=alpha)
    #ax.plot_surface(c.P2[0], c.P2[1], c.P2[2], color=cupColor, alpha=alpha)
    #ax.plot_surface(c.P3[0], c.P3[1], c.P3[2], color=cupColor, alpha=alpha)

    if drawRays:
        draw_rays(p, vuv)
    ax.scatter(xi, yi, zi, s=25, c='k', zorder=10)
    plt.show()


def draw_vuv_to_blue(vuv: np.array, blue: np.array, c : Cylinder, drawVUV=True, drawBlue=True,
                     units=mm, alpha=0.2, barrelColor='blue', cupColor='red',
                     figsize=(16,16)):

    xv,yv,zv = xyz_from_points(vuv)
    xb,yb,zb = xyz_from_points(blue)

    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111, projection='3d')
    draw_cylynder_surface(c, ax, units, alpha, barrelColor, cupColor)

    if drawVUV:
        ax.scatter(xv, yv, zv, s=20, c='k', zorder=10)
    if drawBlue:
        ax.scatter(xb, yb, zb, s=20, c='b', zorder=10)
    plt.show()


def vuv_to_blue_transport_from_point_with_graphics(c: Cylinder, p : np.array,
                                                   scale=1, verbose=True,
                                                   draw=True, figsize=(12,12))->np.array:
    """A VUV photon impinging the cylinder c on point p, is WLS by TPB
    and re-emitted isotropically, then transported to the cylinder walls"""

    def generate_random_point_around_p_inside_cylinder(p : np.array, verbose):
        cnd = True
        cnt = 0
        scale = 1
        while cnd and cnt < 100:
            cnt+=1
            vs = vectors_spherical(npoints=1)
            r = vs[0] + p
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

    def generate_random_point_around_p(p : np.array, verbose):

        vs = scale * vectors_spherical(npoints=1)
        r = vs[0] + p
        if verbose:
            print(f'generating random point {r}')
            pic = point_inside_cylinder(c, r)
            print(f'point inside cylinder? {pic}')

        return r


    def draw_ray(e, d, ax, t):
        tt = np.linspace(0, t, 100)
        xi = e[0] + tt * d[0]
        yi = e[1] + tt * d[1]
        zi = e[2] + tt * d[2]
        if verbose:
            print(f'{xi[0], yi[0], zi[0]}')
            print(f'{xi[-1], yi[-1], zi[-1]}')
        ax.plot(xi, yi, zi)

    if draw:
        fig = plt.figure(figsize=figsize)
        ax=plt.subplot(111, projection='3d')
        draw_cylynder_surface(c, ax)

    #rp   = generate_random_point_around_p_inside_cylinder(p, verbose)
    rp   = generate_random_point_around_p(p, verbose)
    d    = unit_vectors_from_two_points(p,rp)
    r    = Ray(p,d)  # ray from point p in the direction of d
    t, P = ray_intersection_with_cylinder(r, c)

    if verbose:
        print(f'Reflecting from point {p} to point {rp}')
        print(f'Intersection root {t} intersection point {P}')
        print(f'Point in cylynder? {point_in_cylinder(c, P)}')
    if draw:
        ax.scatter(p[0],p[1],p[2], s=25, c='r', zorder=10)
        ax.scatter(rp[0],rp[1],rp[2], s=25, c='b', zorder=10)
        ax.scatter(P[0],P[1],P[2], s=25, c='g', zorder=10)
        draw_ray(p, d, ax, t)

    return P
