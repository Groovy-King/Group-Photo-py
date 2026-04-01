import numpy as np
import astropy.units as u

class Galaxy:
    def __init__(self, pos, properties):
        self.pos = pos
        self.properties = properties
        return