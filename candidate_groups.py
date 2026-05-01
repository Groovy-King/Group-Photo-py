import numpy as np
import pandas as pd
import astropy.units as u
from scipy.stats import shapiro
import argparse

#from Galaxy import Galaxy
#from Group import Group
from constants import c
from loading import load_galaxies
from group_builder import build_groups
from astropy.coordinates import SkyCoord

def identify_groups(galaxies, n_min = 5, p_min = 0.1):
    """
    Threshold for group membership is set to 5 galaxies within the NFW cylinder, and we also require that the velocity distribution 
    of these galaxies is consistent with being drawn from a normal distribution (p-value > 0.1 in the Shapiro-Wilk test)
    """

    gal_list = np.array(list(galaxies), dtype = object)
    groups = set()  # Use a set to store groups to avoid duplicates
    print(f"Loaded {len(galaxies)} galaxies, now identifying groups...")

    # Choose mid points of mass bands
    massspace = np.logspace(10,15,100)
    idxidx = np.round(np.linspace(41,90,10))
    masses = massspace[idxidx.astype(int)]

    # Constructing galaxies position list
    n_galaxies = len(galaxies)
    ra_galaxies = np.empty(n_galaxies, dtype = object)
    dec_galaxies = np.empty(n_galaxies, dtype = object)
    z_galaxies = np.empty(n_galaxies)
    for i, g in enumerate(gal_list):
        ra_galaxies[i] = g.pos[0]
        dec_galaxies[i] = g.pos[1]
        z_galaxies[i] = g.pos[2]
    sky_galaxies = SkyCoord(ra = ra_galaxies, dec = dec_galaxies)

    for galaxy in galaxies:
        # Creating the array of angular separations and redshift differences between the current galaxy and all other galaxies.
        sky_center = SkyCoord(ra = galaxy.pos[0], dec = galaxy.pos[1])
        z_center = galaxy.pos[2]
        angular_separation = sky_center.separation(sky_galaxies).to(u.rad, equivalencies = u.dimensionless_angles()).value
        delta_z = z_galaxies - z_center

        # Build the groups centered on this galaxy for each mass band, and compute the properties needed to define the NFW cylinder.
        build_groups(galaxy, masses)

        

        lowest_mass_group = min(galaxy.core_groups, key = lambda g: g.m500)  # Find the group with the lowest mass among the candidate groups for this galaxy
        lowest_mass_group.NFW_cylinder(angular_separation, delta_z)  
        richness_check = lowest_mass_group.richness >= n_min  # Check if the lowest mass group meets the criteria for minimum richness

        # If the lowest mass group does not meet the criteria for minimum richness, then we can discard all the candidate groups for this galaxy. 
        if richness_check:  
            # Check if the created groups satisfy the criteria for being a candidate group, and if so, 
            # add them to the set of identified groups.
            discarded_groups = set()  # Keep track of groups that do not meet the criteria to avoid redundant checks
            for candidate_group in galaxy.core_groups:
                criteria_check = False
                nfw_mask = candidate_group.NFW_cylinder(angular_separation, delta_z)
                nfw_count = candidate_group.richness
                if nfw_count >= n_min:  # Threshold for group membership
                    galaxy_velocities = np.array([g.pos[2] for g in gal_list[nfw_mask]]) * c.value
                    # Perform the Shapiro-Wilk test for normality on the velocity distribution
                    SW_result = shapiro(galaxy_velocities)
                    if SW_result.pvalue > p_min:  # Threshold for normality
                        groups.add(candidate_group)
                        criteria_check = True

                if criteria_check:
                    candidate_group.add_galaxies(gal_list[nfw_mask])  # Add the galaxies within the NFW cylinder to the group if it meets the criteria for group membership
                else:
                    galaxy.remove_from_groups(set([candidate_group]))  # Remove the candidate group from the galaxy's set of associated groups if it does not meet the criteria for group membership
                    discarded_groups.add(candidate_group)
            
            for dg in discarded_groups:
                galaxy.core_groups.discard(dg)  # Discard the candidate group if it does not meet the criteria for group membership       
        
        else:
            galaxy.remove_from_groups(galaxy.core_groups)  
            galaxy.core_groups.clear() 
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

    filename = f"Data/Galaxies/Vol_{volume}_Slice_{slice}.csv"
    galaxies = load_galaxies(filename)
    groups = identify_groups(galaxies, n_min = 5, p_min = 0.1)
    print(f"Identified {len(groups)} groups in volume {volume} slice {slice}.")