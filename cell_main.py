import numpy as np

from core.Cell.cell import (Cell, Neuron, Astrocyte, Microglia)
from core.point import Point



if __name__ == '__main__':

    
    p1 = Point(0,0)
    print(p1)

    '''
    n1 = Neuron(pos=p1)
    print(n1)
    n1.activate()
    n1.move_to(Point(1,5))
    print(n1)
    '''
    
    m1 = Microglia(pos=p1)
    print(m1)
    m1.activate()
    m1.move_to(Point(1,5))
    print(m1)
    
    print(m1._history)