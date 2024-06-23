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
from typing import Tuple
import numpy as np

def main():
    learning_rate = 1e-6
    batch_size = 2000
    clipnorm = 1
    epochs = 1000
    agent = PolicyAgent.load_policy_agent(File('model'))
    agent.model.summary()
    experience_files = ['experience_13:02:25.965884_10000']
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

    filename = 'trained_model' + datetime.now().time().__str__()
    with File('trained_model_1000', 'w') as updated_agent_outf:
         agent.serialize(updated_agent_outf)

if __name__ == "__main__":
    main()