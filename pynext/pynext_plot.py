import numpy as np
import pandas as pd
import os, sys

from  . system_of_units import *
import matplotlib.pyplot as plt


def set_fonts(ax, fontsize=20):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

def display_figure(x, y, lbx, lby, log=False, xlim=None, ylim=None, xl=None, yl=None,
                   lw=2, fontsize=20, figsize=(8,8)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)

    set_fonts(ax, fontsize=fontsize)
    if log == 'logy':
        plt.semilogy(x, y, linewidth=lw)
    elif log == 'loglog':
        plt.loglog(x, y, linewidth=lw)
    else:
        plt.plot(x, y, linewidth=lw)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if yl:
        plt.axhline(y=yl,linestyle='dashed', color='k',linewidth=lw)
    if xl:
        plt.axvline(x=xl,linestyle='dashed', color='k',linewidth=lw)
    plt.xlabel(lbx)
    plt.ylabel(lby)
    plt.show()


def display_figures(xs, ys, lbx, lby, log=False, xlim=None, ylim=None, xl=None, yl=None,
                   lw=2, fontsize=20, figsize=(8,8)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)

    set_fonts(ax, fontsize=fontsize)
    for i, x in enumerate(xs):
        y = ys[i]
        if log == 'logy':
            plt.semilogy(x, y, linewidth=lw)
        elif log == 'loglog':
            plt.loglog(x, y, linewidth=lw)
        else:
            plt.plot(x, y, linewidth=lw)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if yl:
        plt.axhline(y=yl,linestyle='dashed', color='k',linewidth=lw)
    if xl:
        plt.axvline(x=xl,linestyle='dashed', color='k',linewidth=lw)
    plt.xlabel(lbx)
    plt.ylabel(lby)
    plt.show()
