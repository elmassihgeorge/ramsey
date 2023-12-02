
from Common.game_state import GameState
from typing import Tuple

"""
Encoder interface
"""
class Encoder:
    def __init__(self, order : int = 5):
        self.order = order

    def name(self):
        """
        Retrieve the name of this encoder
        """
        raise NotImplementedError()

    def encode(self, game_state : GameState):
        """
        Encode a game state
        """
        raise NotImplementedError()

    def order(self):
        """
        Returns the graph order
        """
        return self.order

    def shape(self):
        """
        Return the shape of the encoded graph structure
        """
        raise NotImplementedError()
