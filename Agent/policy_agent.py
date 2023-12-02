from Agent.base import Agent
from Common.game_state import GameState
from Common.move import Move
from Encoder.base import Encoder
from keras import Model
import numpy as np
from h5py import File
from Common.util import Util
from Encoder.k4_encoder import K4Encoder

class PolicyAgent(Agent):
    def __init__(self, model: Model, encoder: Encoder):
        self.model = model
        self.encoder = encoder

    def select_move(self, game_state: GameState):
        board_tensor = self.encoder.encode(game_state)
        X = np.array([board_tensor])
        move_probs = self.model.predict(X)[0]
        move_probs = self.clip_probs(move_probs)
        num_edges = self.encoder.order ** 2
        candidates = np.arange(num_edges)
        ranked_edges = np.random.choice(
            candidates, num_edges,
            replace=False, p=move_probs
        )
        print(ranked_edges)
        for edge_idx in ranked_edges:
            edge = self.encoder.decode_edge_index(edge_idx)
            move = Move.play(edge, game_state.active.name)
            if game_state.is_valid_move(move):
                return move
        return Move.pass_turn()
    
    def serialize(self, h5file: File):
        """
        TODO: Store policy agent to disk as h5file
        """
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['order'] = self.encoder.order
        h5file.create_group('model')
        Util.save_model_to_hdf5_group(self.model, h5file['model'])
    
    @classmethod
    def load_policy_agent(cls, h5file: File) -> "PolicyAgent":
        """
        TODO: Load policy agent from h5file
        """
        model = Util.load_model_from_hdf5_group(h5file['model'])
        encoder_name = h5file['encoder'].attrs['name']
        order = h5file['encoder'].attrs['order']
        # TODO: Implement dynamic encoder getting - for now its always k4
        # encoder = Encoder.get_encoder_by_name(encoder_name, order)
        encoder = K4Encoder(order)
        return PolicyAgent(model, encoder)
    
    def clip_probs(self, probs):
        """
        Clips a probability distribution to prevent extrema
        """
        min_p = 1e-5
        max_p = 1 - min_p
        clipped_probs = np.clip(probs, min_p, max_p)
        clipped_probs = clipped_probs / np.sum(clipped_probs)
        return clipped_probs


