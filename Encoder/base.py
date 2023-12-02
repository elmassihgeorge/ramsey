import importlib
from Common.game_state import GameState
from typing import Tuple

"""
Encoder interface
"""
class Encoder:
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
        TODO: Given a (u, v) edge, turn into index in adj matrix
        """
        u, v = edge
        return self.order * u + v
    
    def decode_edge_index(self, index : int) -> Tuple[int, int]:
        """
        TODO: Given an index in adj matrix, turn into (u, v) edge
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
    
    @classmethod
    def get_encoder_by_name(cls, name: str, order: int):  # <1>
        if isinstance(order, int):
            order = order  # <2>
        module = importlib.import_module('Encoder.' + name)
        constructor = getattr(module, '__init__')  # <3>
        return constructor(str(order))
    