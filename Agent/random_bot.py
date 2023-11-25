import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)

import random
from ramsey.Agent.base import Agent
from ramsey.Common.game_state import GameState
from ramsey.Common.move import Move

"""
An agent which selects a random black edge to color
"""
class RandomBot(Agent):
    def select_move(self, game_state : GameState):
        """Choose a random black edge to color
        of the current player's color"""
        player_color = game_state.current_player.name
        black_edges = game_state.board.black_edges()
        return Move.play(random.choice(black_edges), player_color)
    