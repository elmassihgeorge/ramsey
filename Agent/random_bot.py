import random
from Agent.base import Agent
from Common.game_state import GameState
from Common.move import Move

class RandomBot(Agent):
    """
    An agent which selects a random black edge to color
    """
    def select_move(self, game_state : GameState) -> Move:
        """
        Choose a random black edge to color
        of the current player's color
        """
        player_color = game_state.active.name
        black_edges = game_state.board.black_edges()
        if black_edges:
            return Move.play(random.choice(black_edges), player_color)
        else:
            return Move.resign()
    