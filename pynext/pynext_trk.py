import numpy as np
import pandas as pd
import os

from  . system_of_units import *

def read_df(path='/Users/jjgomezcadenas/Projects/Development/databb0nu/',
           filename='NextTon.15atm.bb0nu.0_0_100.df.h5'):
    file = os.path.join(path, 'NextTon.15atm.bb0nu.0_0_100.df.h5')

    return pd.read_hdf(file, 'mcHits')


def track_length_in_z(df, event_id):
    """Returns the track length in z (in mm) for event_id"""
    return np.max(df.loc[event_id].z) - np.min(df.loc[event_id].z)


def track_length_in_steps(df, event_id):
    """Return the track legnth in G4 steps (~0.3 mm) for event_id"""
    return len(df.loc[event_id])


def track_length_in_z_event_interval(df, id_min=0, id_max=100):
    """Track length in z for an event interval"""
    return np.array([track_length_in_z(df, i) for i in range(id_min,id_max)])


def track_length_in_steps_event_interval(df, id_min=0, id_max=100):
    """Track length in steps for an event interval"""
    return np.array([track_length_in_steps(df, i) for i in range(id_min,id_max)])


def rebin_df_in_z(df, event_id=0, nbins=25):
    """Rebins DF in nbins"""
    trk = df.loc[event_id]
    bins = np.linspace(trk.z.min(), trk.z.max(), nbins)
    groups = trk.groupby(pd.cut(trk.z, bins))
    return groups.sum().dropna()


def emax_in_event_interval(df, id_min=0, id_max=100, nbins=25):
    """Returns an array in which each element is the maximum energy deposited per event"""
    RF = [rebin_df_in_z(df, i, nbins) for i in range(id_min,id_max)]
    EMAX = [np.max(rf.E.values/keV) for rf in RF]
    return np.array(EMAX)


def energy_in_event_interval(df, id_min=0, id_max=100, nbins=25):
    """Returns a flat array with the energies of all hits"""
    RF = [rebin_df_in_z(df, i, nbins) for i in range(id_min,id_max)]
    EL = [rf.E.values/keV for rf in RF]
    E = EL[0]
    for e in EL[1:]:
        E = np.concatenate((E,e))
    return E
