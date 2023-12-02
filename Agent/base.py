from Common.game_state import GameState
from Common.move import Move
"""
Interface for a graph coloring agent
"""
class Agent:
    def select_move(self, game_state : GameState) -> Move:
        raise NotImplementedError