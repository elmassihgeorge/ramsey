import importlib
from Common.game_state import GameState
from typing import Tuple

class Encoder:
    """
    Encoder interface
    """
    def __init__(self, order: int):
        self.order = order

    def name(self):
        """
        Retrieve the name of this encoder
        """
        raise NotImplementedError()

    def encode(self, game_state: GameState):
        """
        Encode a game state to a tensor
        """
        raise NotImplementedError()

    def encode_edge(self, edge: Tuple[int, int]) -> int:
        """
        Given a (u, v) edge, turn into index in adj matrix
        """
        u, v = edge
        return self.order * u + v
    
    def decode_edge_index(self, index : int) -> Tuple[int, int]:
        """
        Given an index in adj matrix, turn into (u, v) edge
        """
        return divmod(index, self.order)

    def shape(self):
        """
        Return the shape of the encoded graph structure
        """
        raise NotImplementedError()
    
    def num_edges(self):
        """
        Return the size of the adjaceny matrix of a particular
        """
        return self.order ** 2
