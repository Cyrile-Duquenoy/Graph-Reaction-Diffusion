import numpy as np
import matplotlib.pyplot as plt

class Vertex:
    def __init__(self, ids: int = None, value:float = None):
        self._ids = ids
        self.value = value
        
    
    def __str__(self):
        return f"Vertex(ids={self._ids}, value={self.value})"
    
    def __repr__(self):
        return self.__str__()
    
if __name__ == '__main__':
    v1 = Vertex(ids = 1)
    print(v1)
    v1.value = 0.5
    print(v1)

