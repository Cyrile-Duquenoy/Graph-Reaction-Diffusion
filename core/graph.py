import numpy as np
#import matplotlib.pyplot as plt

from .edge import Edge
from .vertex import Vertex

class Graph:
    def __init__(self, vertices: [Vertex], edges: [Edge]):
        self._vertices = vertices if vertices is not None else np.array([])
        self._edges = edges if edges is not None else []

        self._order = len(self._vertices)
        self._size = len(self._edges)
        
        self.adjacency = self.get_adjacency_matrix()
        self.incidence = self.get_incidence_matrix()
        
        self.laplacian = self.get_laplacian_matrix()
        
        
    def get_vertices(self):
        return self._vertices
    
    def get_edges(self):
        return self._edges
    
    @property
    def order(self):
        return self._order
    
    @property
    def size(self):
        return self._size
    
    def get_adjacency_matrix(self):
        vertices = self.get_vertices()
        edges = self.get_edges()
        n = len(vertices)

        # Map sommet.id -> index 0..n-1
        id_to_idx = {v._ids: idx for idx, v in enumerate(vertices)}

        A = np.zeros((n, n), dtype=int)

        for edge in edges:
            v1, v2 = edge._extremity
            i, j = id_to_idx[v1._ids], id_to_idx[v2._ids]
            A[i, j] = 1
            A[j, i] = 1  # Graphe non orienté, symétriser

        return A
    
    
    def get_incidence_matrix(self):
        vertices = self.get_vertices()
        edges = self.get_edges()
        n = len(vertices)   # Nombre de sommets
        m = len(edges)      # Nombre d'arêtes
        
        id_to_idx = {v._ids: idx for idx, v in enumerate(vertices)}
        I = np.zeros((n, m), dtype=int)
        
        for edge_idx, edge in enumerate(edges):
            v1, v2 = edge._extremity
            i = id_to_idx[v1._ids]
            j = id_to_idx[v2._ids]
            I[i, edge_idx] = 1
            I[j, edge_idx] = 1
            
        return I
    
    def get_laplacian_matrix(self):
        A = self.get_adjacency_matrix()
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        L = D - A
        return L
    
    def get_normal_laplacian_matrix(self):
        adjacency = self.get_adjacency_matrix()
        degree = np.sum(adjacency, axis=1)
        
        D_inv_sqrt = np.zeros((self.order, self.order))
        for i in range(self.order):
            if degree[i] > 0:
                D_inv_sqrt[i, i] = 1.0 / np.sqrt(degree[i])
        
        I = np.eye(self.order)
        normalized_laplacian = I - D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        return normalized_laplacian
    
    '''
    def get_div_matrix(self):
        B = self.get_incidence_matrix()
        u = [edge._ids for edge in self.get_edges()]
        res = B.T @ u
        W = np.eye(self._order)
        J = W @ res
        div = -B @ J
        return div
    '''
    
    def get_div_matrix(self, u, c):
        """
        Calcul de la divergence type Keller–Segel sur un graphe non orienté.
    
        u : np.array, densité sur chaque nœud
        c : np.array, potentiel/champ attractif sur chaque nœud
        """
        n = self._order
        div = np.zeros(n)
        id_to_idx = {v._ids: idx for idx, v in enumerate(self._vertices)}
    
        for edge in self._edges:
            i = id_to_idx[edge._extremity[0]._ids]
            j = id_to_idx[edge._extremity[1]._ids]
    
            # Flux symétrique
            f_edge = 0.5 * (u[i] + u[j]) * (c[j] - c[i])
    
            # Contribution à la divergence
            div[i] += f_edge
            div[j] -= f_edge
    
        return div


    
        
    def __str__(self):
        return f"Graph(vertices={self._vertices}, edges ={self._edges})"

