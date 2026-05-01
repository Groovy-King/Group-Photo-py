import numpy as np
import astropy.units as u
import os
#from Group import Group

class Galaxy:
    """
    A simple class to represent a galaxy with its position and properties. Each galaxy is assigned a unique ID upon initialization.
    """

    # Generate a unique ID for each galaxy instance, while storing the count as a class variable to ensure uniqueness across all instances
    _id_counter = 0
    @classmethod
    def _generate_id(cls):
        pid = os.getpid()  # Get the current process ID to ensure uniqueness across different runs
        id = f"{pid}_{cls._id_counter}"  # Combine process ID with the ID counter
        cls._id_counter += 1
        return id
    
    # Define equality and hashing methods to allow Galaxy instances to be stored in sets and compared based on their unique IDs  
    def __eq__(self, other):
        if not isinstance(other, Galaxy):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    # Constructor to initialize a Galaxy instance with its position (RA, Dec, z) and properties (magnitude, stellar mass, absolute magnitude, 
    # halo mass). The position is stored as a numpy array for easy access and manipulation. 
    def __init__(self, pos, properties, id = None, include_groups = True):
        self.id = id if id is not None else self._generate_id()
        self.pos = pos
        self.properties = properties
        
        if include_groups:
            self.core_groups = set()  # Initialize an empty set to store the groups centered on this galaxy
            self.associated_groups = set()  # Initialize an empty set to store the associated groups that this galaxy belongs to
        return

    def add_to_groups(self, groups):
        """
            Adds this galaxy to the specified groups, and also adds the groups to the galaxy's set of associated groups. 
            This method can be used after identifying candidate groups to populate them with their member galaxies.
        """
        self.associated_groups |= groups  # Add this group to the galaxy's set of associated groups
        for group in groups:
            group.add_galaxies(self, update_galaxies = False)  # Add this galaxy to the group's set of member galaxies
        return
    
    def remove_from_groups(self, groups):
        """
            Removes this galaxy from the specified groups, and also removes the groups from the galaxy's set of associated groups. 
            This method can be used to remove galaxies from groups if they are found to not meet the criteria for group membership.
        """
        self.associated_groups -= groups  # Remove this group from the galaxy's set of associated groups
        for group in groups:
            group.remove_galaxies(self, update_galaxies = False)  # Remove this galaxy from the group's set of member galaxies
        return
    
    def compute_weight(self):
        """
            TO BE IMPLEMENTED: Computes the weight of the probabilistic vote casted by this galaxy, 
            based on its observed luminosity and the luminosity function of galaxies in the universe. 
        """
        pass