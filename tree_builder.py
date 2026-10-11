import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import os
import astropy.units as u
import argparse
import pickle

from util import *
from constants import *
from Galaxy import Galaxy
from Group import Group
from Member_galaxy import MemberGalaxy
from loading import load_galaxies
from candidate_groups import identify_groups

def build_tree(galaxies, extracted_groups):
    """
        Rebuilds the tree structure of groups and their member galaxies from the extracted groups and the original galaxy data.
        This function takes the original galaxy data and the extracted groups, and reconstructs the hierarchical structure of groups and their member galaxies. 
        It returns a list of Group objects, each containing its member galaxies as MemberGalaxy instances.
    """
    groups = set()  # Use a set to avoid duplicate groups
    extracted_groups_list = list(extracted_groups)  # Convert the extracted groups set to a list to maintain order and allow indexing
    galaxies_list = np.array(list(galaxies), dtype = object)  # Convert the galaxies dictionary to a list to maintain order and allow indexing
    # Constructing galaxies position list
    n_galaxies = len(galaxies)
    ra_galaxies = np.empty(n_galaxies, dtype = object)
    dec_galaxies = np.empty(n_galaxies, dtype = object)
    z_galaxies = np.empty(n_galaxies)
    for i, g in enumerate(galaxies_list):
        ra_galaxies[i] = g.pos[0]
        dec_galaxies[i] = g.pos[1]
        z_galaxies[i] = g.pos[2]
    sky_galaxies = SkyCoord(ra = ra_galaxies, dec = dec_galaxies)

    for eg in extracted_groups_list:
        group = Group.from_dict(eg, galaxies_list)  # Create a Group instance from the extracted group dictionary
        group.compute_properties()  # Compute the properties of the group
        
        # Creating the array of angular separations and redshift differences between the current galaxy and all other galaxies.
        sky_center = SkyCoord(ra = group.pos[0], dec = group.pos[1])
        z_center = group.pos[2]
        angular_separation = sky_center.separation(sky_galaxies).to(u.rad, equivalencies = u.dimensionless_angles()).value
        delta_z = z_galaxies - z_center

        # Identify member galaxies of the group based on the NFW cylinder criteria
        nfw_mask = group.NFW_cylinder(angular_separation, delta_z)

        # Building the tree
        group.add_galaxies(galaxies_list[nfw_mask])  # Add the member galaxies to the group
        groups.add(group)

    # Verify that the tree is built correctly by verifying that each group member galaxy has a reference to its parent group and that the group has a list of its member galaxies.
    for group in groups:
        for member in group.member_galaxies:
            assert member.parent_group == group, "Member galaxy does not reference its parent group correctly."
            assert member.parent_group in member.parent_galaxy.associated_groups, "Member galaxy is not properly associated with its parent group."
        assert len(group.member_galaxies) == group.richness, "Group richness does not match the number of member galaxies."
    print("Verified group -> galaxy linking successfully.")

    # Verify that the tree is built correctly by verifying that each group member galaxy has a reference to its parent group and that the group has a list of its member galaxies.
    for galaxy in galaxies:
        for group in galaxy.associated_groups:
            assert galaxy in [mg.parent_galaxy for mg in group.member_galaxies], "Galaxy is not properly associated with its parent group."
    print("Verified galaxy -> group linking successfully.")

    return groups