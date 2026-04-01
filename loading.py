import numpy as np
import pandas as pd
import astropy.units as u

from Galaxy import Galaxy
from Group import Group

def load_galaxies(volume, slice):
    """
        Loads galaxy data from the specified volume and slice, and returns a list of Galaxy objects. The function also checks that the input volume and slice numbers are valid integers within the expected range (volume: 1-9, slice: 1-3). 
        Each Galaxy object is initialized with its position (RA, Dec, z) and properties (magnitude, stellar mass, absolute magnitude, halo mass). Additionally, a boolean attribute 'has_halo' is set for each galaxy based on whether the galaxy 
        has an associated halo in the mock simulation.
    """

    # Check that volume and slice are valid integers within the expected range
    if not isinstance(volume, int) or not isinstance(slice, int):
        raise TypeError("Volume and slice must be integers")
    elif volume < 1 or volume > 9:
        raise ValueError("Volume must be between 1 and 9")
    elif slice < 1 or slice > 3:
        raise ValueError("Slice must be between 1 and 3")
    

    df = pd.read_csv(f'Data/Galaxies/Vol_{volume}_Slice_{slice}.csv')
    galaxies = []
    for i in range(len(df)):
        RA = df.iloc[i, 1] * u.degree
        Dec = df.iloc[i, 2] * u.degree
        z = df.iloc[i, 3]
        m = df.iloc[i, 6] * u.mag
        M_star = df.iloc[i, 7] * u.Msun
        M = df.iloc[i, 8] * u.mag
        M_halo = df.iloc[i, 9] * u.Msun

        center = np.array([RA, Dec, z], dtype=object)

        properties = {
            "m": m,
            "M_star": M_star,
            "M": M,
            "M_halo": M_halo
        }
        gal = Galaxy(center, properties)

        if M_halo > 0:
            gal.has_halo = True
        else:
            gal.has_halo = False
        galaxies.append(gal)

    return np.array(galaxies, dtype = object)