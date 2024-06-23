from Agent.base import Agent
from Common.game_state import GameState
from Common.move import Move
from Encoder.base import Encoder
from keras import Model
import numpy as np
from h5py import File
from Common.util import Util
from Encoder.k3_encoder import K3Encoder
from Encoder.k4_encoder import K4Encoder
from Experience.base import ExperienceCollector
from keras.api.optimizers import SGD

class PolicyAgent(Agent):
    """
    An agent that implements a policy (model)
    """
    def __init__(self, model: Model, encoder: Encoder, collector: ExperienceCollector = None):
        self.model = model
        self.encoder = encoder
        self.collector = collector

    def select_move(self, game_state: GameState):
        """
        Encode a game state and predict a best move using the model
        """
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
        for edge_idx in ranked_edges:
            edge = self.encoder.decode_edge_index(edge_idx)
            move = Move.play(edge, game_state.active.name)
            if game_state.is_valid_move(move):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=edge_idx
                    )
                return move
        return Move.pass_turn()
    
    def train(self, experience, lr, clipnorm, batch_size):
        """
        Train the policy agent's model using experience data
        NOTE: legacy.SGD outperforms SGD on M2 Macbook
        """
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(learning_rate=lr, clipnorm=clipnorm)
        )

        target_vectors = self.prepare_experience_data(
            experience,
            self.encoder.order
        )

        self.model.fit(
            experience.states, target_vectors,
            batch_size=batch_size,
            epochs=1
        )
    
    def serialize(self, h5file: File):
        """
        Store policy agent to disk as h5file
        """
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['order'] = self.encoder.order
        h5file.create_group('model')
        Util.save_model_to_hdf5_group(self.model, h5file['model'])
    
    @classmethod
    def load_policy_agent(cls, h5file: File) -> "PolicyAgent":
        """
        Load policy agent from h5file
        """
        model = Util.load_model_from_hdf5_group(h5file['model'])
        encoder_name = h5file['encoder'].attrs['name']
        order = h5file['encoder'].attrs['order']
        # TODO: Implement dynamic encoder getting - for now its always k3
        # encoder = Encoder.get_encoder_by_name(encoder_name, order)
        encoder = K3Encoder(order)
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
    
    def set_collector(self, collector: ExperienceCollector):
        """
        Set an experience collector
        """
        self.collector = collector

    def prepare_experience_data(self, experience, order):
        """
        Convert experience data into trainable vectors
        """
        experience_size = experience.actions.shape[0]
        target_vectors = np.zeros((experience_size, order ** 2))
        for i in range(experience_size):
            action = experience.actions[i]
            reward = experience.rewards[i]
            target_vectors[i][action] = reward
        return target_vectors
