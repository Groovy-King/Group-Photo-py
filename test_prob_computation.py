import numpy as np
import astropy.units as u
from Galaxy import Galaxy
from astropy.coordinates import SkyCoord
from Group import Group
from Member_galaxy import MemberGalaxy
import pandas as pd

df = pd.read_csv('Data/Galaxies/Vol_1_Slice_1.csv')
galaxies = []

for i in range(1, len(df)):
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

galaxies = np.array(galaxies)

g = galaxies[0]
G = Group(central_galaxy = g, m500 = 1e14 * u.Msun)
G.compute_properties()

mg = MemberGalaxy(g, G)
prob = mg.compute_probability_density()
print(prob)