import numpy as np
import matplotlib.pyplot as plt

from .vertex import Vertex

class Edge:
    def __init__(self, extremity : list[Vertex], ids: int = None, coeff: float = None):     
        if not isinstance(extremity, list) or len(extremity) != 2:
            raise ValueError("extremity must be a list of exactly two Vertex instances.")
        if not all(isinstance(v, Vertex) for v in extremity):
            raise TypeError("Both elements in extremity must be instances of Vertex.")
        
        self._ids = ids if ids else None
        self._extremity = extremity
        self.coeff = coeff
        
    def __str__(self):
        extremities_str = ', '.join(str(v) for v in self._extremity)
        return f"Edge(ids={self._ids}, extremity=[{extremities_str}], coeff={self.coeff})"
    
    def __repr__(self):
        return self.__str__()
    
    



