import numpy as np
import pandas as pd
import astropy.units as u
from scipy.stats import shapiro
import argparse

from Galaxy import Galaxy
from Group import Group
from constants import c
from loading import load_galaxies
from astropy.coordinates import SkyCoord

def identify_groups(filename):
    galaxies = load_galaxies(filename)
    groups = []
    print(f"Loaded {len(galaxies)} galaxies, now identifying groups...")

    # Choose mid points of mass bands
    massspace = np.logspace(10,15,100)
    idxidx = np.round(np.linspace(41,90,10))
    masses = massspace[idxidx.astype(int)]

    # Threshold for group membership is set to 5 galaxies within the NFW cylinder, and we also require that the velocity distribution of these galaxies is consistent with being drawn from a normal distribution (p-value > 0.1 in the Shapiro-Wilk test)
    n_min = 5
    p_min = 0.1

    # Constructing galaxies position list
    ra_galaxies = np.array([g.pos[0] for g in galaxies], dtype = object)
    dec_galaxies = np.array([g.pos[1] for g in galaxies], dtype = object)
    z_galaxies = np.array([g.pos[2] for g in galaxies])
    sky_galaxies = SkyCoord(ra = ra_galaxies, dec = dec_galaxies)

    for galaxy in galaxies:
        sky_group = SkyCoord(ra = galaxy.pos[0], dec = galaxy.pos[1])
        z_group = galaxy.pos[2]

        angular_separation = sky_group.separation(sky_galaxies).to(u.rad, equivalencies = u.dimensionless_angles()).value
        delta_z = z_galaxies - z_group
        for mass in masses:
            candidate_group = Group(galaxy, m500 = mass * u.Msun)
            nfw_mask = candidate_group.NFW_cylinder(angular_separation, delta_z)
            nfw_count = candidate_group.richness
            if nfw_count >= n_min:  # Threshold for group membership
                galaxy_velocities = np.array([g.pos[2] for g in galaxies[nfw_mask]]) * c.value
                # Perform the Shapiro-Wilk test for normality on the velocity distribution
                SW_result = shapiro(galaxy_velocities)
                if SW_result.pvalue > p_min:  # Threshold for normality
                    groups.append(candidate_group)
    return groups

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Identify galaxy groups in a given volume and slice.")
    parser.add_argument("--volume", type = int, default = 1, help = "Volume number (1-9)")
    parser.add_argument("--slice", type = int, default = 1, help = "Slice number (1-3)")
    args = parser.parse_args()

    volume = args.volume
    slice = args.slice

    # Check that volume and slice are valid integers within the expected range
    if not isinstance(volume, int) or not isinstance(slice, int):
        raise TypeError("Volume and slice must be integers")
    elif volume < 1 or volume > 9:
        raise ValueError("Volume must be between 1 and 9")
    elif slice < 1 or slice > 3:
        raise ValueError("Slice must be between 1 and 3")

    groups = identify_groups(f"Data/Galaxies/Vol_{volume}_Slice_{slice}.csv")
    print(f"Identified {len(groups)} groups in volume {volume} slice {slice}.")