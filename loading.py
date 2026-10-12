import numpy as np
import pandas as pd
import astropy.units as u

from Galaxy import Galaxy
from Group import Group
from True_group import TrueGroup

def load_galaxies(filename):
    """
        Loads galaxy data from the specified file, and returns a list of Galaxy objects. The function also checks that the input volume and slice numbers are valid integers within the expected range (volume: 1-9, slice: 1-3). 
        Each Galaxy object is initialized with its position (RA, Dec, z) and properties (magnitude, stellar mass, absolute magnitude, halo mass). Additionally, a boolean attribute 'has_halo' is set for each galaxy based on whether the galaxy 
        has an associated halo in the mock simulation.
    """
    
    df = pd.read_csv(filename)
    galaxies = set()  # Use a set to store galaxies to avoid duplicates
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
            "apparent_mag": m,
            "stellar_mass": M_star,
            "absolute_mag": M,
            "halo_mass": M_halo
        }
        gal = Galaxy(center, properties)

        if M_halo > 0:
            gal.has_halo = True
        else:
            gal.has_halo = False
        galaxies.add(gal)

    return galaxies

def load_true_groups(filename, galaxies):
    """
        Loads true group data from the specified file, and returns a list of Group objects. Each Group object is initialized with its position (RA, Dec, z) and properties (richness, r200, sigma_z). 
        The function also checks that the input volume and slice numbers are valid integers within the expected range (volume: 1-9, slice: 1-3).
    """

    df = pd.read_csv(filename)
    true_groups = set()  # Use a set to store true groups to avoid duplicates
    n_groups = max(df.iloc[:, 0])  # Get the maximum group ID to determine the number of groups
    for i in range(n_groups):
        group_data = df[df.iloc[:, 0] == i]
        if not group_data.empty:
            # Initialize the group with its position and properties
            members = set()
            for j in range(len(group_data)):
                gal = next((g for g in galaxies if g.pos[0] == group_data.iloc[j, 1] * u.degree and g.pos[1] == group_data.iloc[j, 2] * u.degree and g.pos[2] == group_data.iloc[j, 3]), None)
                if gal is not None:
                    members.add(gal)
                else:
                    raise ValueError(f"Warning: Galaxy with position ({group_data.iloc[j, 1]}, {group_data.iloc[j, 2]}, {group_data.iloc[j, 3]}) not found in the loaded galaxies.")

            halo_masses = [group_data.iloc[j, -1] for j in range(len(group_data))]
            if len(np.unique(halo_masses)) > 1:
                raise ValueError(f"Warning: Inconsistent halo masses for group {i}.")
            else:
                m500 = halo_masses[0] * u.Msun
            TG = TrueGroup(members, m500)
            true_groups.add(TG)
    return true_groups