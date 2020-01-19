import numpy as np
import pandas as pd
import os, sys

from scipy.linalg import norm

from  . system_of_units import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from . pynext_types import  Cylinder, Sphere
from . pynext_base  import  sample_spherical
from . pynext_base  import  xyz_from_vuv


def set_fonts(ax, fontsize=20):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


def draw_cylinder(c : Cylinder, alpha=0.2, barrelColor='blue', cupColor='red',
                  figsize=(16,16)):


    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111, projection='3d')
    ax.plot_surface(c.P[0], c.P[1], c.P[2], color=barrelColor, alpha=alpha)
    ax.plot_surface(c.P2[0], c.P2[1], c.P2[2], color=cupColor, alpha=alpha)
    ax.plot_surface(c.P3[0], c.P3[1], c.P3[2], color=cupColor, alpha=alpha)

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


def draw_vuv(vuv: np.array, c : Cylinder, p=np.array([0,0,0]), alpha=0.2, barrelColor='blue', cupColor='red', drawRays=True,
                  figsize=(16,16)):

    def draw_rays(e, R):

        for r in R:
            tt = np.linspace(0, 2*r[3], 100)
            xi = e[0] + tt * r[0]
            yi = e[1] + tt * r[1]
            zi = e[2] + tt * r[2]
            ax.plot(xi, yi, zi)

    xi,yi,zi = xyz_from_vuv(vuv)

    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111, projection='3d')
    ax.plot_surface(c.P[0], c.P[1], c.P[2], color=barrelColor, alpha=alpha)
    ax.plot_surface(c.P2[0], c.P2[1], c.P2[2], color=cupColor, alpha=alpha)
    ax.plot_surface(c.P3[0], c.P3[1], c.P3[2], color=cupColor, alpha=alpha)

    if drawRays:
        draw_rays(p, vuv)
    ax.scatter(xi, yi, zi, s=25, c='k', zorder=10)
    plt.show()
