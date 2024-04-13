import numpy as np
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer

def function_OU(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)[0]

class NeuralAgent():
    def __init__(self):
        BUFFER_SIZE = 100000
        TAU = 0.001 # Target network hyperparameters
        LRA = 0.0001 # Learning rate for actor
        LRC = 0.001 # Learning rate for critic
        state_dim = 3 # # of sensors input
        self.batch_size = 32
        self.lambda_mix = 10.0
        self.action_dim = 1

        self.actor = ActorNetwork(state_dim, self.action_dim, TAU, LRA)
        self.critic = CriticNetwork(state_dim, self.action_dim, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE) # Create replay buffer
