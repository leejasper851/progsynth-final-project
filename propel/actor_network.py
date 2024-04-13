import torch.nn as nn
import torch

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork:
    def __init__(self, state_size, action_size, TAU, LEARNING_RATE):
        self.TAU = TAU

        # Now create the model
        self.model = self.create_actor_network(state_size, action_size)
        self.target_model = self.create_actor_network(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    
    def train(self, states, action_grads):
        self.model.train()
        params_grad = torch.autograd.grad(self.model(torch.from_numpy(states)), self.model.parameters(), grad_outputs=tuple([-e for e in action_grads]))
        self.optimizer.zero_grad()
        for param, param_grad in zip(self.model.parameters(), params_grad):
            param.grad = param_grad
        self.optimizer.step()
    
    def target_train(self):
        actor_weights = self.model.parameters()
        actor_target_weights = self.target_model.parameters()
        for weight, target_weight in zip(actor_weights, actor_target_weights):
            target_weight.data = self.TAU * weight + (1 - self.TAU) * target_weight

    def create_actor_network(self, state_size, action_dim):
        print("Now we build the model")
        return ActorModel(state_size, action_dim)

class ActorModel(nn.Module):
    def __init__(self, state_size, action_dim):
        super().__init__()

        self.h0 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.h0_relu = nn.ReLU()
        self.h1 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h1_relu = nn.ReLU()
        self.V = nn.Linear(HIDDEN2_UNITS, action_dim)
        nn.init.trunc_normal_(self.V.weight, mean=0.0, std=0.0001, a=-0.0003, b=0.0003)
        self.V_tanh = nn.Tanh() # TODO: make activation more general
    
    def forward(self, S):
        S = self.h0_relu(self.h0(S))
        S = self.h1_relu(self.h1(S))
        return 2 * self.V_tanh(self.V(S))
    
    def predict(self, S): # TODO: figure out when to train() vs. eval() and no_grad()
        # self.eval()
        return self(S)
