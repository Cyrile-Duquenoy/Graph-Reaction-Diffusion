import numpy as np

from core.Cell.cell import (Cell, Neuron, Astrocyte, Microglia)
from core.point import Point


from core.edge import Edge
from core.vertex import Vertex
from core.graph import Graph

from core.cellgraph import CellGraph

# Création des cellules
cells = [Neuron(pos=Point(0,0)), Astrocyte(pos=Point(1,0)), Microglia(pos=Point(1.5,2))]

# Création du graphe
cg = CellGraph(cells, connect_rule="fully_connected")

# Initialiser densité
u = cg.set_density()

# Champ attractif aléatoire
c = np.random.rand(len(cells))

# Calcul divergence
div = cg.compute_divergence(u, c)
print(div)

# Visualisation
cg.plot()

lap = cg.graph.get_laplacian_matrix()
print(lap)



