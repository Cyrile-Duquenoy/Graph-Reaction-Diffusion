import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x:float, y: float, z:float = None):
        self.x = x
        self.y = y
        self.z = z
        if z is None:
            self.coord = [x, y]
        else:
            self.coord = [x, y, z]
        
    def __str__(self):
        if self.z is not None:
            return f'Point(x={self.x},y={self.x},z={self.z})'
        else:
            return f'Point(x={self.x},y={self.x})'
    
    def dist(self, p:'Point'):
        res = 0
        for i in range(len(self.coord)):
            res += (self.coord[i] - p.coord[i])**2
        return np.sqrt(res)
    

if __name__ == '__main__':
    p1 = Point(0,0,1)
    print(p1)
    p2 = Point(0,0,1)
    print(p1.dist(p2))
    print(p2.coord)
