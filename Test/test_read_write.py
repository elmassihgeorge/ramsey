import sys, os


file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)
from Encoder.k4_encoder import K4Encoder
import h5py
from Agent.policy_agent import PolicyAgent
from Common.board import Board
from Common.game_state import GameState
from Encoder.k3_encoder import K3Encoder
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, InputLayer

def main():
    ORDER = 5
    NUM_EDGES = ORDER ** 2
    CLIQUE_ORDERS = (3, 3)
    encoder = K4Encoder(ORDER)
    model = Sequential()
    model.add(InputLayer(input_shape=encoder.shape()))
    model.add(Dense(encoder.num_planes*NUM_EDGES**2, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(NUM_EDGES**2, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(NUM_EDGES, activation='softmax'))
    model.summary()
    policy_agent = PolicyAgent(model, encoder)

    # Serialize policy agent to an h5 file:
    with h5py.File("my_first_model", 'w') as outf:
        policy_agent.serialize(outf)

    # Reconstruct a policy agent:
    my_new_agent = PolicyAgent.load_policy_agent(h5py.File('my_first_model'))

    # Produce a move from this new agent:
    game_state = GameState(Board(ORDER), CLIQUE_ORDERS)
    print(my_new_agent.select_move(game_state))

if __name__ == "__main__":
    main()