import numpy as np
import astropy.units as u

class Galaxy:
    """
    A simple class to represent a galaxy with its position and properties. Each galaxy is assigned a unique ID upon initialization.
    """

    # Generate a unique ID for each galaxy instance, while storing the count as a class variable to ensure uniqueness across all instances
    _id_counter = 0
    @classmethod
    def _generate_id(cls):
        i = cls._id_counter
        cls._id_counter += 1
        return i

    def __init__(self, pos, properties):
        self.id = self._generate_id()
        self.pos = pos
        self.properties = properties
        return