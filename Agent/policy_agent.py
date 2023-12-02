from Agent.base import Agent
from Common.game_state import GameState
from Encoder.base import Encoder
from keras import Model

class PolicyAgent(Agent):
    def __init__(self, model : Model, encoder : Encoder):
        self.model = model
        self.encoder = encoder

    def select_move(self, game_state : GameState):
        raise NotImplementedError


