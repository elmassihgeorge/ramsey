from collections import namedtuple
import sys, os

file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)
from Agent.random_bot import RandomBot
from Common.game_state import GameState
from Agent.base import Agent
from Common.player import Player
from Common.util import Util
from Experience.base import ExperienceBuffer, ExperienceCollector
from Agent.policy_agent import PolicyAgent
from h5py import File
from typing import Tuple
import numpy as np


class GameRecord(namedtuple('GameRecord', 'moves win')):
    pass


def main():
    red_agent = PolicyAgent.load_policy_agent(File('trained_model_1000'))
    blue_agent = PolicyAgent.load_policy_agent(File('trained_model_1000'))
    red_wins = 0
    red_losses = 0
    blue_wins = 0
    blue_losses = 0
    num_games = 10000
    order = 5
    clique_orders = (3, 3)
    color1 = Player.red
    for i in range(num_games):
        game_record = simulate_game(red_agent, blue_agent, order, clique_orders)
        print('Simulating Game %d/%d Simulated ... [%r]' % (i + 1, num_games, game_record.win))
        if Player.red in game_record.win:
            red_wins += 1
        else:
            red_losses += 1
        if Player.blue in game_record.win:
            blue_wins += 1
        else:
            blue_losses += 1
        color1 = color1.other
    print('Red Agent Record: %d/%d' % (red_wins, red_wins + red_losses))
    print('Blue Agent Record: %d/%d' % (blue_wins, blue_wins + blue_losses))


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
    main()
