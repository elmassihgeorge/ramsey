import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)
from Common.game_state import GameState
from Agent.base import Agent
from Common.player import Player
from Common.util import Util
from Experience.base import ExperienceBuffer, ExperienceCollector
from Agent.policy_agent import PolicyAgent
from h5py import File
from typing import Tuple
import numpy as np

def main():
    learning_rate = 1e-9
    batch_size = 4096
    clipnorm = 1
    epochs = 1000
    agent = PolicyAgent.load_policy_agent(File('12_4_23_model'))
    agent.model.summary()
    experience_files = ['experience_12_4_2023_100000']
    for i in range(epochs):
        print('Training epoch %s...' % (i + 1))
        for exp_filename in experience_files:
            exp_buffer = ExperienceBuffer.load_experience(File(exp_filename))
            agent.train(
                exp_buffer,
                lr=learning_rate,
                clipnorm=clipnorm,
                batch_size=batch_size
            )

    with File('trained_12_4_23_model_1000', 'w') as updated_agent_outf:
         agent.serialize(updated_agent_outf)

if __name__ == "__main__":
    main()