import sys, os
from datetime import datetime

file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)
from Encoder.k4_encoder import K4Encoder
import h5py
from Agent.policy_agent import PolicyAgent
from Common.board import Board
from Common.game_state import GameState
from Encoder.k3_encoder import K3Encoder
from keras import Sequential
from keras.api.layers import Dense, Dropout, Flatten, InputLayer


def main():
    ORDER = 5
    NUM_EDGES = ORDER ** 2
    CLIQUE_ORDERS = (3, 3)
    encoder = K3Encoder(ORDER)
    model = Sequential()
    model.add(InputLayer(shape=encoder.shape()))
    model.add(Dense(108, activation='sigmoid'))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(108, activation='sigmoid'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(NUM_EDGES, activation='softmax'))
    policy_agent = PolicyAgent(model, encoder)

    # Serialize policy agent to an h5 file:
    filename = "model_" + datetime.now().time().__str__()
    with h5py.File(filename, 'w') as outf:
        policy_agent.serialize(outf)

    # Reconstruct a policy agent:
    my_new_agent = PolicyAgent.load_policy_agent(h5py.File(filename))

    # Produce a move from this new agent:
    game_state = GameState(Board(ORDER), CLIQUE_ORDERS)
    print(my_new_agent.select_move(game_state))


if __name__ == "__main__":
    main()
