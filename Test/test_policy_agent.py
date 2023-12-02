import sys, os
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)
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
    encoder = K3Encoder(ORDER)
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
    game_state = GameState(Board(ORDER), CLIQUE_ORDERS)
    print(policy_agent.select_move(game_state))

if __name__ == "__main__":
    main()