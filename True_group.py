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