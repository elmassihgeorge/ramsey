import sys, os



file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)
from Common.game_state import GameState
from Agent.base import Agent
from Common.player import Player
from Common.util import Util
from Experience.base import ExperienceBuffer, ExperienceCollector
import Experience

from Common.view import View
from Agent.policy_agent import PolicyAgent
from h5py import File
from typing import Tuple
import numpy as np

def main():
    agent1 = PolicyAgent.load_policy_agent(File('my_first_model'))
    agent2 = PolicyAgent.load_policy_agent(File('my_second_model'))
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    num_games = 1
    for i in range(num_games):
        collector1.begin_episode()
        collector2.begin_episode()
        winners = simulate_game(agent1, agent2, 5, (3, 3))
        print("Winner:", i, winners)
        if len(winners) == 0:
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=-1)
        if Player.red in winners:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        if Player.blue in winners:
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)

    experience = combine_experience([collector1, collector2])
    with File('my_second_experience', 'w') as experience_outf:
        experience.serialize(experience_outf)

def simulate_game(agent_1: "Agent", agent_2: "Agent", order: int, clique_orders: Tuple[int, int]):
        game = GameState.new_game(order, clique_orders)
        agents = {
            Player.red: agent_1,
            Player.blue: agent_2
        }
        while not game.is_over():
            next_move = agents[game.active].select_move(game)
            print("Move:", next_move)
            game = game.apply_move(next_move)
        View(game.board).render()
        return game.winners()

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