from Common.game_state import GameState
from Common.move import Move
from typing import Tuple

from Common.player import Player
class Agent:
    """
    Interface for a graph coloring agent
    """
    def select_move(self, game_state : GameState) -> Move:
        """
        Select an edge to color
        """
        raise NotImplementedError
