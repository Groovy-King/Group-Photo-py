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

def proximity_merging(Groups):
    """
        First phase of the post-processing steps, we sort the groups based on descending order of their central galaxy luminosity, check for other group centers within their NFW cylinder,
        and merge all nearby groups into the most luminous seeding group. This functions takes in the sorted array of groups and the list of galaxies, and returns the merged groups.
    """
    groups = np.copy(Groups)  # Create a copy of the groups array to avoid modifying the original
    sky_groups = SkyCoord(ra = [g.pos[0] for g in groups], dec = [g.pos[1] for g in groups])
    z_groups = np.array([g.pos[2] for g in groups])
    merged_groups = set()  # List to store the merged groups


    for group in groups:  # Loop until there are no more groups left to process
        if not hasattr(group, 'hough_index'):
            raise AttributeError("Group objects must be assigned a hough index before merging.")
        if not hasattr(group, 'rank'):
            raise AttributeError("Group objects must be ranked in descending order of luminosity.")
        if hasattr(group, 'proximity_merged'):
            continue  # Skip this group if it has already been merged into another group

        # For each group, we check for other group centers within its NFW cylinder
        sky_center = SkyCoord(ra = group.pos[0], dec = group.pos[1])
        z_center = group.pos[2]
        angular_separation = sky_center.separation(sky_groups).to(u.rad, equivalencies = u.dimensionless_angles()).value
        delta_z = z_groups - z_center

        nfw_mask = group.NFW_cylinder(angular_separation, delta_z)
        nearby_groups = groups[nfw_mask]  # Get the nearby groups within the NFW cylinder
        nearby_groups = nearby_groups[[not hasattr(g, 'proximity_merged') for g in nearby_groups]]  # Filter out the nearby groups that have already been merged into another group
        if len(nearby_groups) == 0:
            merged_groups.add(group)  # If there are no nearby groups, then we can add this group to the set of merged groups as is
            continue

        # If there are nearby groups, we identify the central group with the highest Hough index among the nearby groups, and merge all nearby groups into this central group
        central_group = nearby_groups[np.argmax([g.hough_index for g in nearby_groups])]  # Identify the central group with the highest Hough index among the nearby groups
        
        # We then merge all nearby groups into the central group
        for nearby_group in nearby_groups:
            if nearby_group != central_group and not hasattr(nearby_group, 'proximity_merged'):  # Check if this nearby group is not the central group itself and has not been merged into another group yet
                galaxies = set()
                for galaxy in nearby_group.member_galaxies:
                    galaxies.add(galaxy.parent_galaxy)  # Add the member galaxies of this nearby group to the set of galaxies to be merged into the central group
                central_group.add_galaxies(galaxies)  # Add the member galaxies of the nearby group to the central group, and creates the bidirectional links between the member galaxies and the central group
                nearby_group.remove_galaxies(galaxies)  # Remove the member galaxies from the nearby group, essentially deleting the nearby group as it has been merged into the central group
                nearby_group.proximity_merged = True  # Mark this nearby group as merged to avoid redundant checks in future iterations

        merged_groups.add(central_group)  # Add the central group (which now contains the merged nearby groups) to the set of merged groups
        group.proximity_merged = True  # Mark this group as merged to avoid redundant checks in future iterations

    return merged_groups


def membership_merging(Groups):
    """
        Second phase of the post-processing steps, we check groups for common member galaxies, and merge the groups that share atleast half the members of the smaller group. 
        This function takes the results of the first step of post processing as input (input is assumed to be sorted in descending order of BCG luminosity), and returns the final merged groups.

        This method currently checks every possible pair of groups for membership merging, which is computationally expensive but ensures that we do not miss any potential merges. 
        However, this can be optimized in the future by defining a more efficient way to identify nearby groups that are likely to share members, and only checking those pairs for membership merging.
    """
    n = len(Groups)
    merged_groups = set()  # List to store the merged groups

    for i in range(n):
        group = Groups[i]
        current_members = group.member_galaxies  # Get the member galaxies of this group
        if i == n - 1 and not hasattr(group, 'membership_merged'):
            merged_groups.add(group)  # If this is the last group, we can add it to the set of merged groups as is
            group.membership_merged = True  # Mark this group as merged to avoid redundant checks in future iterations
            continue
        if hasattr(group, 'membership_merged'):
            continue  # Skip this group if it has already been merged into another group

        groups = Groups[i + 1:]  # Get the groups that come after this group to check for membership merging
        for other_group in groups: 
            other_members = other_group.member_galaxies  # Get the member galaxies of the other group
            # Check if the groups share at least half the members of the smaller group
            n_threshold = min(len(current_members), len(other_members)) / 2  # Define the threshold for membership merging as half the number of members in the smaller group
            if len(current_members.intersection(other_members)) >= n_threshold:
                galaxies = set()
                for galaxy in other_group.member_galaxies:
                    galaxies.add(galaxy.parent_galaxy)  # Add the member galaxies of this nearby group to the set of galaxies to be merged into the central group
                group.add_galaxies(galaxies)  # Add the member galaxies of the other group to the current group
                other_group.remove_galaxies(galaxies)  # Remove the member galaxies from the other group, essentially deleting the other group as it has been merged into the current group
                other_group.membership_merged = True

        group.membership_merged = True  # Mark this group as merged to avoid redundant checks in future iterations
        merged_groups.add(group)
    return merged_groups