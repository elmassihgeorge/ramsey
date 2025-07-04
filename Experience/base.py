from h5py import File
import numpy as np

class ExperienceBuffer:
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def serialize(self, h5file: File):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)

    @classmethod
    def load_experience(cls, h5file: File):
        return ExperienceBuffer(
            states=np.array(h5file['experience']['states']),
            actions=np.array(h5file['experience']['actions']),
            rewards=np.array(h5file['experience']['rewards'])
        )
    
class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_episode_states = []
        self.current_episode_actions = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []

    def record_decision(self, state, action):
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)

    def complete_episode(self, reward, custom_rewards=None):
        num_states = len(self.current_episode_actions)
        self.states += self.current_episode_states
        self.actions += self.current_episode_actions
        
        if custom_rewards is None:
            # Current behavior - uniform rewards
            self.rewards += [reward for _ in range(num_states)]
        else:
            # Allow for step-specific rewards
            assert len(custom_rewards) == num_states, \
                f"custom_rewards length {len(custom_rewards)} doesn't match number of states {num_states}"
            self.rewards += custom_rewards

        self.current_episode_states = []
        self.current_episode_actions = []

    def to_buffer(self):
        return ExperienceBuffer(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards)
        )
    