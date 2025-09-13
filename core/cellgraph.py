import numpy as np
import matplotlib.pyplot as plt
from .vertex import Vertex
from .edge import Edge
from .graph import Graph
from .Cell.cell import Neuron, Astrocyte, Microglia

class CellGraph:
    def __init__(self, cells: list, connect_rule="linear"):
        """
        cells : liste d'objets Cell (Neuron, Astrocyte, Microglia)
        connect_rule : str, règle de connexion ("linear", "fully_connected", etc.)
        """
        self.cells = cells
        self.vertices = [Vertex(ids=cell._global_id, value=0.0) for cell in cells]
        self.edges = self._create_edges(connect_rule)
        self.graph = Graph(vertices=self.vertices, edges=self.edges)
        #self.positions = {v._ids: (np.random.rand(), np.random.rand()) for v in self.vertices}
        self.positions = {v._ids: (cells[idx].pos[0], cells[idx].pos[1]) for idx, v in enumerate(self.vertices)}


    def _create_edges(self, rule):
        edges = []
        if rule == "linear":
            # Connecte chaque cellule à sa voisine
            for i in range(len(self.vertices) - 1):
                edges.append(Edge(extremity=[self.vertices[i], self.vertices[i+1]], coeff=1.0))
        elif rule == "fully_connected":
            # Graphe complet
            for i in range(len(self.vertices)):
                for j in range(i+1, len(self.vertices)):
                    edges.append(Edge(extremity=[self.vertices[i], self.vertices[j]], coeff=1.0))
        # Ajouter d'autres règles si nécessaire
        return edges

    def set_density(self, u_dict=None):
        """
        u_dict : dict {Cell : value} ou None pour initialiser selon le type
        """
        u = np.zeros(len(self.vertices))
        if u_dict is None:
            # initialisation par type
            for idx, cell in enumerate(self.cells):
                if isinstance(cell, Neuron):
                    u[idx] = 1.0
                elif isinstance(cell, Astrocyte):
                    u[idx] = 0.5
                elif isinstance(cell, Microglia):
                    u[idx] = 0.2
        else:
            # initialisation personnalisée
            for idx, cell in enumerate(self.cells):
                u[idx] = u_dict.get(cell, 0.0)
        return u

    def compute_divergence(self, u, c):
        return self.graph.get_div_matrix(u, c)
    
    def update_graph(self, connect_rule="fully_connected"):
        """
        Recalcule les arêtes du graphe selon les nouvelles positions.
        """
        self.edges = self._create_edges(connect_rule)
        self.graph = Graph(vertices=self.vertices, edges=self.edges)
        
        
    def update_positions(self, dt=0.05, bounds=(0, 3)):
        min_x, max_x = bounds
        for idx, cell in enumerate(self.cells):
            if isinstance(cell, Microglia):
                dx, dy = np.random.randn(2) * dt
                cell.pos[0] += dx
                cell.pos[1] += dy
    
                # Rester dans les bornes définies
                cell.pos[0] = np.clip(cell.pos[0], min_x, max_x)
                cell.pos[1] = np.clip(cell.pos[1], min_x, max_x)
    
                self.positions[self.vertices[idx]._ids] = (cell.pos[0], cell.pos[1])


    def plot(self):
        # Tracer les arêtes
        for edge in self.edges:
            x = [self.positions[edge._extremity[0]._ids][0], self.positions[edge._extremity[1]._ids][0]]
            y = [self.positions[edge._extremity[0]._ids][1], self.positions[edge._extremity[1]._ids][1]]
            plt.plot(x, y, 'k-', alpha=0.5)

        # Tracer les sommets avec couleur selon type
        for idx, cell in enumerate(self.cells):
            pos = self.positions[self.vertices[idx]._ids]
            if isinstance(cell, Neuron):
                plt.scatter(pos[0], pos[1], c='red', s=100, label='Neuron' if idx == 0 else "")
            elif isinstance(cell, Astrocyte):
                plt.scatter(pos[0], pos[1], c='blue', s=100, label='Astrocyte' if idx == 0 else "")
            elif isinstance(cell, Microglia):
                plt.scatter(pos[0], pos[1], c='green', s=100, label='Microglia' if idx == 0 else "")

        plt.legend()
        plt.show()


