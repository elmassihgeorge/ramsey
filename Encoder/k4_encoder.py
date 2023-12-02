from Common.game_state import GameState
from Encoder.base import Encoder
import numpy as np
import networkx
"""
Encoder that includes information about 3 and 4 cliques:
Plane 0: Black edges
Plane 1: Red edges
Plane 2: Blue edges
Plane 3: Red 3-cliques
Plane 4: Blue 3-cliques
Plane 5: Red 4-cliques
Plane 6: Blue 4-cliques
"""
class K4Encoder(Encoder):
    def __init__(self, order : int):
        self.order = order
        self.num_planes = 7

    def name():
        return 'k4_encoder'
    
    def encode(self, game_state : GameState):
        """
        Return an adjacency matrix of game_state.board
        """
        board = game_state.board
        black_subgraph = board.get_monochromatic_subgraph("black")
        red_subgraph = board.get_monochromatic_subgraph("red")
        blue_subgraph = board.get_monochromatic_subgraph("blue")

        black_adjacency_matrix = networkx.to_numpy_array(black_subgraph)
        red_adjacency_matrix = networkx.to_numpy_array(red_subgraph)
        blue_adjacency_matrix = networkx.to_numpy_array(blue_subgraph)

        red_k3 = board.get_monochromatic_clique_subgraph("red", 3)
        red_k3_adjacency_matrix = networkx.to_numpy_array(red_k3)
        blue_k3 = board.get_monochromatic_clique_subgraph("blue", 3)
        blue_k3_adjacency_matrix = networkx.to_numpy_array(blue_k3)

        red_k4 = board.get_monochromatic_clique_subgraph("red", 4)
        red_k4_adjacency_matrix = networkx.to_numpy_array(red_k4)
        blue_k4 = board.get_monochromatic_clique_subgraph("blue", 4)
        blue_k4_adjacency_matrix = networkx.to_numpy_array(blue_k4)
        
        return np.array([black_adjacency_matrix, red_adjacency_matrix, blue_adjacency_matrix,
                         red_k3_adjacency_matrix, blue_k3_adjacency_matrix,
                         red_k4_adjacency_matrix, blue_k4_adjacency_matrix])
    
    def shape(self):
        return self.num_planes, self.order, self.order
