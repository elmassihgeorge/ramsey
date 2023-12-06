import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)

import time
from Common.view import View
from Common.game_state import GameState
from Common.player import Player
from Agent.random_bot import RandomBot
from typing import Tuple
import Agent
from Test.eval_agent import GameRecord

def random_game():
    """
    Two random bot opponents attempt to color a K_5 graph
    without forming red or blue triangles
    """
    ORDER = 5
    CLIQUE_ORDERS = (3, 3)
    wins = 0
    losses = 0
    num_games = 1000
    for i in range(num_games):
        game_record = simulate_game(RandomBot(), RandomBot(), ORDER, CLIQUE_ORDERS)
        if game_record.win:
            wins += 1
        else:
            losses +=1
    print('Random Agent Record: %d/%d' % (wins, wins + losses))

def simulate_game(agent_1: "Agent", agent_2: "Agent", order: int, clique_orders: Tuple[int, int]):
    game = GameState.new_game(order, clique_orders)
    moves = []
    agents = {
        Player.red: agent_1,
        Player.blue: agent_2
    }
    while not game.is_over():
        next_move = agents[game.active].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
    return GameRecord(moves=moves, win=game.win())

if __name__ == "__main__":
    random_game()