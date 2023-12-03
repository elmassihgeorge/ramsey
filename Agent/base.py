from Common.game_state import GameState
from Common.move import Move
from typing import Tuple

from Common.player import Player
"""
Interface for a graph coloring agent
"""
class Agent:
    def select_move(self, game_state : GameState) -> Move:
        raise NotImplementedError
    
    @classmethod
    def simulate_game(agent_1: "Agent", agent_2: "Agent", order: int, clique_orders: Tuple[int, int]):
        game = GameState.new_game(order, clique_orders)
        agents = {
            Player.red: agent_1,
            Player.blue: agent_2
        }
        while not game.is_over():
            next_move = agents[game.player].select_move(game)
            game = game.apply_move(next_move)
        return game.winners()