from Common.game_state import GameState
from Encoder.base import Encoder
import numpy as np
import networkx

""" Encoder implementation to convert a game state into an np.array adjacency matrix
Plane 0: Black edges
Plane 1: Red edges
Plane 2: Blue edges
"""
class TwoColorEncoder(Encoder):
    def __init__(self, order : int):
        self.order = order
        self.num_planes = 3

    def name():
        return 'two_color_encoder'
    
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
        return np.array([black_adjacency_matrix, red_adjacency_matrix, blue_adjacency_matrix])
    
    def shape(self):
        return self.num_planes, self.order, self.order

        