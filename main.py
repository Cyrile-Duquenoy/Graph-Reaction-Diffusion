import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from core import(Vertex,
                 Edge,
                 Graph)
# ___________________________________________________________________________ #

# Coefficient de diffusion
dk = 1.0

# Discrétisation temportelle
T = 100
M = 100
dt = 1 / (M - 1)
t = np.linspace(0,T,M)


if __name__ == '__main__':
    
    v1 = Vertex(ids=1, value=1.0)
    v2 = Vertex(ids=2, value=1.0)
    v3 = Vertex(ids=3, value=2.0)
    
    e1 = Edge(extremity=[v1, v2], ids=1, coeff=dk)
    e2 = Edge(extremity=[v2, v3], ids=2, coeff=dk)
    e3 = Edge(extremity=[v1,v3], ids=3, coeff=dk)
    
    '''Construction d'un graphe étoilé'''
    G = Graph([v1, v2, v3], [e1, e2,e3])
    print(G, '\n')
    

    '''
    Choisir si on travaille sur le graphe 1 ou 2
    '''
    lap = G.get_laplacian_matrix()
    print('laplacien : ',lap, '\n')
    
    
    # Liste de noeuds et d'arrêtes
    vertices = [v1, v2, v3]
    edges = [e1, e2, e3]
    
    u = [vertex.value for vertex in vertices]
    
    # Conservation de la masse en stationnaire
    '''
    On cherche le vecteur propre associé à la valeur propre nulle
    '''
    eigvals, eigvecs = np.linalg.eig(lap)
    idx = np.argmin(np.abs(eigvals))
    stationnaire = eigvecs[:, idx].real
    stationnaire *= sum(u) / sum(stationnaire)
    print('Solution stationnaire : ', stationnaire, '\n')
    print('Conservation de la masse en stationnaire : ', sum(stationnaire), '\n')
    
    
#%% Diffusion Pure

    '''################'''
    ### Diffusion Pure ###
    '''################'''
    
    # Coefficient de diffusion
    ## Diffusion isotrope donc matrice diagonale de forme dk*Id
    D = np.eye(len(u))*1.0
    
    ## Itération en temps
    U_over_time = []
    V=[]
    
    U=[] # Stockage des normes de u
    S=[] # Stockage de la masse totale
    

    for i in range(len(t)):
        du = -dt * D @ lap @ u      # variation diffusion seule
        u = u + du                  # mise à jour de u
        
        # Stockage pour animation
        U_over_time.append(u.copy())
        
        # Norme et masse totale
        res = np.linalg.norm(u)
        s = np.sum(u)
        U.append(res)
        S.append(s)
        
        # Stockage des variations instantanées
        V.append(np.linalg.norm(lap @ u))  # v = L u
        
        
    '''
    PLOT
    '''
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 ligne, 2 colonnes
    
    # Etude de convergence vers etat stationnaire
    axes[0].plot(t, V)
    axes[0].set_title("Convergence vers état stationnaire")
    axes[0].set_xlabel("temps")
    axes[0].set_ylabel("||L @ u||")
    axes[0].set_yscale('log')
    
    # Conservation de la masse
    axes[1].plot(t, S)
    axes[1].set_title("Masse totale de u")
    axes[1].set_xlabel("temps")
    axes[1].set_ylabel("sum(u)")
    
    # Titre général
    fig.suptitle("Évolution de u dans le modèle de diffusion / Keller–Segel", fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Affichage final des valeurs de u
    print("u final :", u)
    print("Masse totale :", np.sum(u))
    
    # Animation
    
    fig, ax = plt.subplots(figsize=(6,4))
    bar_container = ax.bar(range(len(u)), U_over_time[0], tick_label=[f"v{i+1}" for i in range(len(u))])
    ax.set_ylim(0, max([max(u_t) for u_t in U_over_time]) * 1.1)
    ax.set_ylabel("Valeur de u")
    ax.set_xlabel("Noeuds")
    ax.set_title("Évolution de u sur les noeuds")
    
    def animate(frame):
        for bar, height in zip(bar_container, U_over_time[frame]):
            bar.set_height(height)
        return bar_container
    
    anim = FuncAnimation(fig, animate, frames=len(U_over_time), interval=50, blit=False)
    plt.show()
    
    
#%% Reaction-Diffusion type Keller-Segel
    
    
    '''################'''
    ### Reaction-Diffusion ### 
    ### type Keller-Segel  ###
    '''################'''

    
    v1 = Vertex(ids=1, value=1.0)
    v2 = Vertex(ids=2, value=1.0)
    v3 = Vertex(ids=3, value=1.0)
    
    e1 = Edge(extremity=[v1, v2], ids=1, coeff=dk)
    e2 = Edge(extremity=[v2, v3], ids=2, coeff=dk)
    e3 = Edge(extremity=[v1,v3], ids=3, coeff=dk)
    
    vertices = [v1, v2, v3]
    edges = [e1, e2, e3]
    
    '''Construction d'un graphe étoilé'''
    G = Graph([v1, v2, v3], [e1, e2, e3])
    print(G, '\n')
    '''Pour un graphe en chaine : utiliser le grpahe ci dessous'''
    # G = Graph([v1, v2, v3], [e1, e2])
    
    lap = G.get_normal_laplacian_matrix()
    
    u = [vertex.value for vertex in vertices]

    # Pas sûr si la formulation de la divergence est bonne (c.f. classe Graphe)
    c = np.array([0.1, 0.5, 0.5])
    
    
    div = G.get_div_matrix(u, c)
    print(div)
    
    
    U_list = []  # norme de u
    S_list = []  # masse totale
    V_list = []
    VV_list = []
    
    # iteration en temps
    
    for i in range(len(t)):
        # Calcul du flux Keller-Segel
        # divergence chimio-attractif
    
        # diffusion + flux
        u = u + dt * (- D @ lap @ u + div)
    
        U_list.append(np.linalg.norm(u))
        S_list.append(np.sum(u))
        
        # Stockage des variations instantanées
        V_list.append(np.linalg.norm(- D @ lap @ u + div))  # v = L u
        VV_list.append(np.linalg.norm(lap @ u))
    
    '''
    PLOT
    '''
    
    fig = plt.figure(figsize=(10,6))

    # --- Grille 2x2 mais on fusionne la 2ème ligne ---
    gs = fig.add_gridspec(2, 2)
    
    # Première ligne → deux sous-plots
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    
    # Deuxième ligne → un seul subplot qui prend toute la largeur
    ax3 = fig.add_subplot(gs[1,:])

    # --- Premier plot : norme de u ---
    ax1.plot(t, V_list)
    ax1.set_title("Norme de - D @ lap @ u + div")
    ax1.set_xlabel("temps")
    ax1.set_ylabel("||- D @ lap @ u + div||")

    # --- Premier plot : Domination Diffusion / Reaction ---
    
    '''
    Interprétation possible :
        Courbe décrosisante --> Diffusion dominante
        Courbe croissante --> Réaction dominante
    '''
    
    ax2.plot(t, VV_list)
    ax2.set_title("Domination Diffusion / Reaction")
    ax2.set_xlabel("temps")
    ax2.set_ylabel("|| L @ u ||")
    
    # --- Premier plot : norme de u ---
    ax3.plot(t, S_list)
    ax3.set_title("Conservation de la masse")
    ax3.set_xlabel("temps")
    ax3.set_ylabel("sum(u)")
    
    fig.suptitle("Évolution du système Keller-Segel sur graphe", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Affichage final des valeurs de u
    print("u final :", u)
    print("Masse totale :", np.sum(u))
     
    
    
    
    
    
    
