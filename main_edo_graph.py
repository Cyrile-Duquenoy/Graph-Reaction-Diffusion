import numpy as np
import matplotlib.pyplot as plt
from core.Cell.cell import Cell, Neuron, Astrocyte, Microglia
from core.point import Point
from core.cellgraph import CellGraph

# Création des cellules
cells = [
    Neuron(pos=Point(0, 0)),
    Neuron(pos=Point(1, 0)),
    Astrocyte(pos=Point(2, 0)),
    Microglia(pos=Point(0, 1)),
    Microglia(pos=Point(1.5, 2)),
]

cg = CellGraph(cells, connect_rule="fully_connected")

# Initialisation
u = cg.set_density()
D = np.eye(len(u)) * 0.05

# Champ attractif (fixe pour l'instant)
c = np.random.rand(len(cells))

# Temps
T = 10
M = 200
dt = T / M
t = np.linspace(0, T, M)

# Stockage
U_over_time = []
Positions_over_time = []

for step in range(M):
    # 1. Mettre à jour les positions des cellules mobiles
    cg.update_positions(dt=0.05, bounds=(0, 3))
    
    # Stocker les positions actuelles
    current_positions = {v._ids: (cell.pos[0], cell.pos[1]) for v, cell in zip(cg.vertices, cg.cells)}
    Positions_over_time.append(current_positions)


    # 2. Recalculer graphe (voisinages et matrices)
    cg.update_graph(connect_rule="fully_connected")

    # 3. Recalculer le Laplacien
    lap = cg.graph.get_laplacian_matrix()

    # 4. Diffusion
    du_diff = -dt * D @ lap @ u

    # 5. Réaction locale (exemples simples)
    du_reac = np.zeros_like(u)
    for idx, cell in enumerate(cg.cells):
        if isinstance(cell, Neuron):
            du_reac[idx] = dt * (0.1 * u[idx] * (1 - u[idx]))  # Croissance logistique
        elif isinstance(cell, Astrocyte):
            du_reac[idx] = dt * (-0.05 * u[idx])               # Dégradation
        elif isinstance(cell, Microglia):
            du_reac[idx] = dt * (0.02 * c[idx])                # Réponse à c

    # 6. Mise à jour de u
    u += du_diff + du_reac
    U_over_time.append(u.copy())

# Animation / Visualisation
# (tu peux réutiliser ta fonction d’animation ici)



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6, 6))

# Initialisation des points et arêtes
scat = ax.scatter([], [], s=[], c=[], cmap='viridis', vmin=0, vmax=np.max(U_over_time), edgecolor='k')
lines = []

def init():
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    return scat,

def update(frame):
    ax.clear()

    # Récupérer les positions et valeurs à cet instant
    u_frame = U_over_time[frame]
    #positions = {v._ids: (cells[idx].pos[0], cells[idx].pos[1]) for idx, v in enumerate(cg.vertices)}
    positions = Positions_over_time[frame]


    # Tracer les arêtes
    for edge in cg.edges:
        id1, id2 = edge._extremity[0]._ids, edge._extremity[1]._ids
        x_vals = [positions[id1][0], positions[id2][0]]
        y_vals = [positions[id1][1], positions[id2][1]]
        ax.plot(x_vals, y_vals, 'gray', alpha=0.4)

    # Tracer les cellules
    pos_array = np.array(list(positions.values()))
    color_array = u_frame
    size_array = 100 + 300 * (u_frame / np.max(U_over_time))  # Taille proportionnelle à u

    scat = ax.scatter(pos_array[:, 0], pos_array[:, 1], s=size_array, c=color_array,
                      cmap='viridis', vmin=0, vmax=np.max(U_over_time), edgecolor='k')

    ax.set_title(f"t = {frame*dt}")
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    return scat,

# Créer l'animation
anim = FuncAnimation(fig, update, frames=len(U_over_time), init_func=init, blit=False, interval=50)

plt.show()


