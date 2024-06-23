import sys, os
from datetime import datetime

file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)
from Common.game_state import GameState
from Agent.base import Agent
from Common.player import Player
from Common.util import Util
from Experience.base import ExperienceBuffer, ExperienceCollector
from Agent.policy_agent import PolicyAgent
from h5py import File
from typing import Tuple, List
import numpy as np

def main():
    red_agent = PolicyAgent.load_policy_agent(File('model_12:37:46.667155'))
    blue_agent = PolicyAgent.load_policy_agent(File('model_12:37:46.667155'))
    red_collector = ExperienceCollector()
    blue_collector = ExperienceCollector()
    red_agent.set_collector(red_collector)
    blue_agent.set_collector(blue_collector)

    num_games = 10000
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        red_collector.begin_episode()
        blue_collector.begin_episode()
        winners = simulate_game(red_agent, blue_agent, 5, (3, 3))
        if Player.red in winners:
            red_collector.complete_episode(reward=1)
        else:
            red_collector.complete_episode(reward=-1)
        if Player.blue in winners:
            blue_collector.complete_episode(reward=1)
        else:
            blue_collector.complete_episode(reward=-1)
        print(f"Game {i} Winners: {winners}")

    experience = combine_experience([red_collector, blue_collector])
    filename = 'experience_' + datetime.now().time().__str__() + '_' + num_games.__str__()
    with File(filename, 'w') as experience_outf:
        experience.serialize(experience_outf)

def simulate_game(agent_1: "Agent", agent_2: "Agent", order: int, clique_orders: Tuple[int, int]) -> List:
        game = GameState.new_game(order, clique_orders)
        agents = {
            Player.red: agent_1,
            Player.blue: agent_2
        }
        while not game.is_over():
            next_move = agents[game.active].select_move(game)
            game = game.apply_move(next_move)
        return game.win()

def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])

    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_rewards)

if __name__ == "__main__":
    main()