import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from Group import Group
from Member_galaxy import MemberGalaxy

class TrueGroup(Group):
    """
    A subclass of Group that represents a true group of galaxies. 
    This class inherits from the Group class and can have additional properties or methods specific to true groups if needed. 
    """
    def __init__(self, member_galaxies, m500):
        """
            Takes a set of Galaxy instances that are members of the true group and the M500 mass of the group.
            The constructor identifies the most massive galaxy (MMG) in the group and initializes the parent Group class with the MMG's position and the member galaxies. It also sets the parent true group reference for each member galaxy.
        """
        member_galaxies = np.array(list(member_galaxies))  # Ensure member galaxies are stored as a list
        galaxy_masses = np.array([g.properties["stellar_mass"].value for g in member_galaxies]) 
        MMG = member_galaxies[np.argmax(galaxy_masses)]  # Identify the most massive galaxy (MMG) in the group
        self._MMG = MMG  # Store the MMG as an attribute of the TrueGroup instance
        super().__init__(MMG, m500, add_central_galaxy = False)  # Initialize the parent Group class with the MMG's position and the member galaxies

        # Add member galaxies to the group and set their parent group reference
        members = set()
        for galaxy in member_galaxies:
            # Check if the galaxy already belongs to another TrueGroup to avoid conflicts
            if hasattr(galaxy, 'parent_true_group') and galaxy.parent_true_group is not None:
                raise ValueError(f"Galaxy {galaxy} already belongs to a TrueGroup.")
            member = MemberGalaxy(galaxy, self)
            members.add(member)
            galaxy.parent_true_group = self  # Set the parent true group reference for the original Galaxy instance
        self._member_galaxies |= members  # Use set union to add galaxies to the group, ensuring no duplicates

        self.primary_extracted_group = None  # Initialize the primary extracted group reference to None, to be set later when the true group is associated with an extracted group
        return

    # Define equality and hashing methods to allow TrueGroup instances to be stored in sets and compared based on their unique IDs  
    def __eq__(self, other):
        if not isinstance(other, Group):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    # Override the add_galaxy method to ensure that TrueGroup instances cannot add/remove galaxies after initialization, as true groups are fixed by definition
    def add_galaxies(self):
        raise NotImplementedError("TrueGroup instances cannot add/remove galaxies after initialization.")

    def remove_galaxies(self):
        raise NotImplementedError("TrueGroup instances cannot add/remove galaxies after initialization.")

    def add_galaxy(self):
        raise NotImplementedError("TrueGroup instances cannot add/remove galaxies after initialization.")

    def remove_galaxy(self):
        raise NotImplementedError("TrueGroup instances cannot add/remove galaxies after initialization.")

    def compute_empirical_prior(self):
        raise NotImplementedError("TrueGroup instances do not compute empirical priors, as they are fixed by definition.")

    # Override the MMG() property to return the most massive galaxy in the true group
    @property
    def MMG(self):
        """
            Returns the most massive galaxy (MMG) in the true group. 
            This property is computed during initialization and stored as an attribute of the TrueGroup instance.
        """
        return self._MMG

    # Setup linking between TrueGroup and its associated extracted group based on MMG and membership
    # MMG Linking
    def mmg_linking(self, associated_extracted_groups):
        """
            WARNING: This method is intended to be called after the TrueGroup and its member galaxies have been initialized, and after the extracted groups tree has been rebuilt.
            
            Links the TrueGroup to its associated extracted group based on the MMG. 
            This method checks if the MMG of the TrueGroup is also the MMG of any extracted group, and if so, establishes a link between them.
            Since the link can only be established to one of the associated extracted groups of the MMG, we do not need to pass all the extracted groups as an argument, as they can be accessed through the MMG's associated groups.
            However, since the set of associated groups depends of the group finder used, the argument still must be passed to the method to ensure that the correct set of associated extracted groups is used for linking.
        """
        out_dict_self = dict()
        out_dict_eg = dict()
        if not associated_extracted_groups:
            out_dict_self["identification"] = "False Negative"  # MMG is not associated with any extracted group
            out_dict_self["primary_extracted_group"] = None  # No primary extracted group can be linked
        else:
            candidate_primary_extracted_groups = list()
            for group in associated_extracted_groups:
                if group.MMG == self.MMG:
                    candidate_primary_extracted_groups.append(group)

            if len(candidate_primary_extracted_groups) == 1:
                out_dict_self["primary_extracted_group"] = candidate_primary_extracted_groups[0]
                out_dict_self["identification"] = "True Positive"  # MMG is correctly identified in the extracted group

                out_dict_eg["primary_true_group"] = self
                out_dict_eg["identification"] = "True Positive"  

            elif len(candidate_primary_extracted_groups) > 1:
                out_dict_self["primary_extracted_group"] = max(candidate_primary_extracted_groups, key = lambda g: g.m500)  # Choose the group with the highest m500 as the primary extracted group
                out_dict_self["identification"] = "True Positive"  # MMG is correctly identified in multiple extracted groups
                out_dict_self["special_case"] = True  # Special case where multiple extracted groups have the same MMG

                out_dict_eg["primary_true_group"] = self
                out_dict_eg["identification"] = "True Positive"

            else:
                out_dict_self["primary_extracted_group"] = None  # No matching extracted group found for the MMG
                out_dict_self["identification"] = "False Negative"  # MMG is not correctly identified in any extracted group
        return out_dict_self, out_dict_eg