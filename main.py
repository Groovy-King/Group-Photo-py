import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
import argparse
import pickle
from scipy.stats import shapiro
from pathlib import Path

from Galaxy import Galaxy
from Group import Group
from constants import *
from loading import load_galaxies
from candidate_groups import identify_groups
from util import *
from post_processing import proximity_merging, membership_merging

parser = argparse.ArgumentParser()

parser.add_argument("--volume", type = int, default = 1, help = "Volume index to process (1-9)")
parser.add_argument("--slice", type = int, default = 1, help = "Slice index to process (1-3)")
parser.add_argument("--save_output", type = bool, default = True, help = "Whether to save the output pickle files containing the extracted groups")

args = parser.parse_args()
volume = args.volume
slice = args.slice
save_output = args.save_output


# Check that volume and slice are valid integers within the expected range
if not isinstance(volume, int) or not isinstance(slice, int):
    raise TypeError("Volume and slice must be integers")
elif volume < 1 or volume > 9:
    raise ValueError("Volume must be between 1 and 9")
elif slice < 1 or slice > 3:
    raise ValueError("Slice must be between 1 and 3")

# Load galaxies using the given file
filename = f"Data/Galaxies/Vol_{volume}_Slice_{slice}.csv"
galaxies = load_galaxies(filename)
gal_list = np.array(list(galaxies), dtype = object)

# Identify groups using the loaded galaxies
groups = identify_groups(galaxies, n_min = 5, p_min = 0.1)
print(f"Identified {len(groups)} groups in volume {volume} slice {slice}.")

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
print("Constructed galaxy position list.")

# Compute the necessary probabilities and weights for the identified groups required to cast the probabilistic vote (Hough index)
# No. 1: Positional probability density (Also computes the velocity dispersion of the group using the gapper estimator)
for group in groups:
    member_galaxies = group.member_galaxies  # Get the member galaxies of the group
    velocities = []
    for member in member_galaxies:
        member.compute_probability_density()  # Compute the probability density for each member galaxy based on its position relative to the group center
        velocities.append(c * member.pos[2])  

    velocities = np.array(velocities, dtype = object)  # Convert the list of velocities to a numpy array for easier manipulation    
    group.sigma_v_gapper = gapper_estimator(velocities)  # Compute the velocity dispersion of the group using the gapper estimator
    group.compute_m500_gapper()  # Compute the mass of the group using the gapper estimator of velocity dispersion
print("Computed positional probability density and velocity dispersion for each group.")

# No. 2: Empirical Prior Probability
Group.compute_empirical_prior(groups)  # Compute the empirical prior probability for each group based on its richness and the distribution of groups in its redshift band
print("Computed empirical prior probability for each group.")

# No. 3: Mass band Probability
for galaxy in galaxies:
    galaxy.compute_mass_band_probability()  # Compute the mass band probability for groups centered on this galaxy based on their richness and the distribution of groups in the same redshift band
print("Computed mass band probability for each group.")

# No. 4: Weight of the probabilistic vote casted by each galaxy
Galaxy.compute_weight(galaxies, beta = -0.4)  
print("Computed weight of the probabilistic vote casted by each galaxy.")

# Compute the Hough index for each group based on the probabilities and weights computed above.
for group in groups:
    member_galaxies = group.member_galaxies  # Get the member galaxies of this group
    for member in member_galaxies:
        member.hough_vote = member.probability_density_positional * group.probability_prior_empirical * member.parent_galaxy.weight  # Compute the Hough vote of this member galaxy for this group based on the positional probability density of this member galaxy for this group, the empirical prior probability of this group, and the weight of the probabilistic vote casted by this member galaxy
        if hasattr(member.parent_galaxy, 'hough_votes_non_normalized'):
            member.parent_galaxy.hough_votes_non_normalized.append(member.hough_vote)  # Store the non-normalized Hough vote of this member galaxy for this group in the parent galaxy instance for later normalization
        else:
            member.parent_galaxy.hough_votes_non_normalized = [member.hough_vote]  # Initialize the list to store the non-normalized Hough votes of this member galaxy for this group in the parent galaxy instance for later normalization

for galaxy in galaxies:
    if hasattr(galaxy, 'hough_votes_non_normalized'):
        galaxy.hough_normalization_factor = sum(galaxy.hough_votes_non_normalized)  # Compute the normalization factor by summing the non-normalized Hough votes of this galaxy for all associated groups

for group in groups:
    member_galaxies = group.member_galaxies  # Get the member galaxies of this group
    for member in member_galaxies:
        member.hough_vote_normalized = member.hough_vote / member.parent_galaxy.hough_normalization_factor  # Normalize the Hough vote of this member galaxy for this group by dividing it by the normalization factor computed for the parent galaxy instance     

# Build the extracted groups, and compute the Hough index for each group based on the probabilities and weights computed above.
extracted_groups = set()  # Initialize an empty set to store the extracted groups
for galaxy in galaxies:
    if galaxy.core_groups:  # Check if this galaxy is a core galaxy of any group
        m500 = 0 * u.Msun  # Initialize the mass of the group to be extracted from this galaxy to be zero
        hough_index = 0  # Initialize the Hough index of the group to be extracted from this galaxy to be zero
        for group in galaxy.core_groups:
            m500 += group.m500_gapper * group.probability_mass_band
            hough_index += group.probability_mass_band * np.sum([member.hough_vote_normalized for member in group.member_galaxies])  # Compute the Hough index of the extracted group by summing the Hough votes of the member galaxies, weighted by the mass band probability of this group

        # Create a new Group instance for the extracted group, using this galaxy as the central galaxy and the computed mass.
        # We follow the whole retinue of initialization steps for the Group class, including computing properties, but building the NFW cylinder and member galaxy instances, and the new bidirectional links will be built later once all the extracted groups are in place
        extracted_group = Group(galaxy, m500)  
        extracted_group.hough_index = hough_index  # Assign the computed Hough index to this extracted group
        extracted_group.compute_properties()  
        extracted_groups.add(extracted_group)  # Add this extracted group to the set of extracted groups
print(f"Extracted {len(extracted_groups)} groups and assigned Hough indices.")

# Before building the extracted group memberships, we clear all the associated groups from each galaxy
for galaxy in galaxies:
    galaxy.remove_from_groups(galaxy._associated_groups)  # Clear all the associated groups from this galaxy to prepare for building the extracted group memberships

# Build the NFW cylinder and member galaxy instances for each extracted group, and establish the bidirectional links between the member galaxies and the extracted group.
for group in extracted_groups:
    sky_center = SkyCoord(ra = group.pos[0], dec = group.pos[1])
    z_center = group.pos[2]
    angular_separation = sky_center.separation(sky_galaxies).to(u.rad, equivalencies = u.dimensionless_angles()).value
    delta_z = z_galaxies - z_center

    nfw_mask = group.NFW_cylinder(angular_separation, delta_z)
    group.add_galaxies(gal_list[nfw_mask])  # Add the galaxies within the NFW cylinder to the group, and creates the bidirectional links between the member galaxies and this group
print("Built the NFW cylinder and member galaxy instances for each extracted group, and established the bidirectional links between the member galaxies and the extracted groups.")

extracted_groups_sorted = list(extracted_groups)
extracted_groups_sorted.sort(key = lambda g: g.central_galaxy.properties["absolute_mag"], reverse = False)  # Sort the extracted groups based on the luminosity of their central galaxy in descending order
for index, group in enumerate(extracted_groups_sorted):
    group.rank = index + 1  # Assign the rank of this group based on the sorted order, with the most luminous central galaxy being rank 1
print("Sorted the extracted groups based on the luminosity of their central galaxy.")

proximity_merged_groups = proximity_merging(extracted_groups_sorted)  # Perform proximity merging on the extracted groups based on their Hough indices and the NFW cylinders
print(f"Performed proximity merging and reduced the number of groups from {len(extracted_groups)} to {len(proximity_merged_groups)}.")

# Convert the set of proximity merged groups to a list and sort it based on the luminosity of the central galaxy in descending order before performing membership merging, 
# as the membership merging function assumes that the input groups are sorted in descending order of luminosity
proximity_merged_groups_sorted = list(proximity_merged_groups)
proximity_merged_groups_sorted.sort(key = lambda g: g.central_galaxy.properties["absolute_mag"], reverse = False)  # Sort the proximity merged groups based on the luminosity of their central galaxy in descending order
membership_merged_groups = membership_merging(proximity_merged_groups_sorted)  # Perform membership merging on the proximity merged groups based on their shared member galaxies
print(f"Performed membership merging and reduced the number of groups from {len(proximity_merged_groups)} to {len(membership_merged_groups)}.")

# Check for Shopiro-Wilk condition on the final merged groups, and discard the groups that do not satisfy the condition
final_groups = set()  # Initialize an empty set to store the final groups after discarding the groups that do not satisfy the Shapiro-Wilk condition
for group in membership_merged_groups:
    galaxy_velocities = np.array([g.pos[2] for g in group.member_galaxies]) * c.value  # Get the velocities of the member galaxies of this group
    SW_result = shapiro(galaxy_velocities)  # Perform the Shapiro-Wilk test for normality on the velocity distribution of the member galaxies of this group
    if SW_result.pvalue > 0.1:  # Check if the p-value from the Shapiro-Wilk test is greater than the threshold for normality
        final_groups.add(group)  # If the velocity distribution of the member galaxies of this group satisfies the Shapiro-Wilk condition, then we can add this group to the set of final groups
print(f"Checked the Shapiro-Wilk condition and reduced the number of groups from {len(membership_merged_groups)} to {len(final_groups)}.")

if save_output:
    BASE_DIR = Path(__file__).resolve().parent
    output_dir = BASE_DIR / "Output"
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / f"Extracted_Groups_Vol_{volume}_Slice_{slice}.pkl"

    # To avoid recursion issues, we rebuild the final groups with only the necessary information for later analysis, which includes the position, mass, richness, and the member galaxies.
    #saveable_galaxies = load_galaxies(filename)
    saveable_final_groups = [{'pos': group.pos, 
                            'm500': group.m500,
                            } for group in final_groups]
        #saveable_group.compute_properties()  # Compute the properties of this group based on the central galaxy and mass information
        #saveable_group.search_for_members(saveable_galaxies)  # Search for member galaxies for this group using the same criteria as before, but using the saveable galaxies loaded from the file to avoid recursion issues

    pickle.dump(saveable_final_groups, open(file_path, "wb"))  # Save the extracted groups to a pickle file for later analysis
