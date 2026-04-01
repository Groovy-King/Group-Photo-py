import numpy as np
import pandas as pd
import astropy.units as u
from scipy.stats import shapiro

from Galaxy import Galaxy
from Group import Group
from constants import c
from loading import load_galaxies

def identify_group_centers(volume, slice):
    galaxies = load_galaxies(volume, slice)
    group_centers = []

    # Choose mid points of mass bands
    massspace = np.logspace(10,15,100)
    idxidx = np.round(np.linspace(41,90,10))
    masses = massspace[idxidx.astype(int)]

    # Threshold for group membership is set to 5 galaxies within the NFW cylinder, and we also require that the velocity distribution of these galaxies is consistent with being drawn from a normal distribution (p-value > 0.1 in the Shapiro-Wilk test)
    n_min = 5
    p_min = 0.1

    for mass in masses:
        for galaxy in galaxies:
            candidate_group = Group(galaxy, {"M500": mass * u.Msun})
            nfw_mask = candidate_group.NFW_cylinder(galaxies)
            nfw_count = np.sum(nfw_mask)
            
            if nfw_count >= n_min:  # Threshold for group membership
                galaxy_velocities = np.array([g.pos[2]*c for g in galaxies[nfw_mask]])
                # Perform the Shapiro-Wilk test for normality on the velocity distribution
                SW_result = shapiro(galaxy_velocities)
                if SW_result.pvalue > p_min:  # Threshold for normality
                    group_centers.append(candidate_group)
    return group_centers