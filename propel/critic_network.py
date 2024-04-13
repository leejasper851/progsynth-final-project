import torch.nn as nn
import torch

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork():
    def __init__(self, state_size, action_size, TAU, LEARNING_RATE):
        self.TAU = TAU

        # Now create the model
        self.model = self.create_critic_network(state_size, action_size, LEARNING_RATE)
        self.target_model = self.create_critic_network(state_size, action_size, LEARNING_RATE)

    def gradients(self, states, actions):
        self.model.train()
        return torch.autograd.grad(self.model(torch.from_numpy(states), actions), actions, grad_outputs=torch.ones(len(states), 1))

    def target_train(self):
        critic_weights = self.model.parameters()
        critic_target_weights = self.target_model.parameters()
        for weight, target_weight in zip(critic_weights, critic_target_weights):
            target_weight.data = self.TAU * weight + (1 - self.TAU) * target_weight

    def create_critic_network(self, state_size, action_dim, LEARNING_RATE):
        print("Now we build the model")
        return CriticModel(state_size, action_dim, LEARNING_RATE)

class CriticModel(nn.Module):
    def __init__(self, state_size, action_dim, LEARNING_RATE):
        super().__init__()

        self.w1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.w1_relu = nn.ReLU()
        self.a1 = nn.Linear(action_dim, HIDDEN2_UNITS)
        self.h1 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h3 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.h3_relu = nn.ReLU()
        self.V = nn.Linear(HIDDEN2_UNITS, action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, S, A):
        S = self.w1_relu(self.w1(S))
        A = self.a1(A)
        S = self.h1(S)
        SA = S + A
        SA = self.h3_relu(self.h3(SA))
        return self.V(SA)
    
    def predict(self, S, A):
        # self.eval()
        return self(S, A)
    
    def train_on_batch(self, S, A, y_t):
        self.train()
        loss = self.loss_fn(self(S, A), y_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
