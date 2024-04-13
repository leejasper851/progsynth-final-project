import numpy as np
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer
import gymnasium as gym
import logging
import utils
import torch
import copy

def function_OU(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)[0]

class NeuralAgent():
    def __init__(self):
        self.ENV_NAME = "Pendulum-v1"
        self.STATE_DIMS = 3
        self.ACTION_DIMS = 1
        self.MAX_EPISODE_LEN = 200
        self.ACTION_MIN = (-2,)
        self.ACTION_MAX = (2,)
        self.BEST_VAL_IND = 0
        self.BEST_VAL_NAME = "X"

        BUFFER_SIZE = 100 * self.MAX_EPISODE_LEN
        TAU = 0.001 # Target network hyperparameters
        LRA = 0.0001 # Learning rate for actor
        LRC = 0.001 # Learning rate for critic
        self.batch_size = 32
        self.lambda_mix = 10.0

        self.actor = ActorNetwork(self.STATE_DIMS, self.ACTION_DIMS, TAU, LRA)
        self.critic = CriticNetwork(self.STATE_DIMS, self.ACTION_DIMS, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE) # Create replay buffer

    def update_neural(self, controllers, episode_count=200, tree=False):
        GAMMA = 0.99
        EXPLORE = 200.0 * self.MAX_EPISODE_LEN
        max_steps = 2 * self.MAX_EPISODE_LEN
        done = False
        epsilon = 1
        min_epsilon = 0.01

        if not tree:
            action_prog = controllers[0]
        
        # Generate an environment
        env = gym.make(self.ENV_NAME)

        window = 5
        lambda_store = np.zeros((max_steps, 1))
        lambda_max = 40.0
        factor = 0.8

        logging.info(f"{self.ENV_NAME} experiment start with Lambda {self.lambda_mix}")

        for i_episode in range(episode_count):
            logging.info(f"Episode {i_episode}")
            logging.info(f"Replay Buffer {self.buff.count()}")
            logging.info("")
            ob, _ = env.reset()

            s_t = np.hstack(ob)

            total_reward = 0.0
            temp_obs = [[ob[i]] for i in range(len(ob))] + [[0]]
            window_list = [temp_obs[:] for _ in range(window)]

            for j_iter in range(max_steps):
                if tree:
                    tree_obs = [sensor for obs in temp_obs[:-1] for sensor in obs]
                    act_tree = controllers.predict([tree_obs])
                    action_action = utils.clip_to_range(act_tree[0][0], self.ACTION_MIN[0], self.ACTION_MAX[0])
                else:
                    action_action = utils.clip_to_range(action_prog.pid_execute(window_list), self.ACTION_MIN[0], self.ACTION_MAX[0])
                
                action_prior = [action_action]

                temp_obs = [[ob[i]] for i in range(len(ob))] + [action_prior]
                window_list.pop(0)
                window_list.append(temp_obs[:])

                epsilon -= 1.0 / EXPLORE
                epsilon = max(epsilon, min_epsilon)
                a_t = np.zeros([1, self.ACTION_DIMS])
                noise_t = np.zeros([1, self.ACTION_DIMS])

                a_t_original = self.actor.model.predict(torch.from_numpy(s_t.reshape(1, len(s_t)))).detach().numpy()
                noise_t[0][0] = max(epsilon, 0) * function_OU(a_t_original[0][0], 0.0, 0.15, 0.2)

                a_t[0][0] = a_t_original[0][0] + noise_t[0][0]

                mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(self.ACTION_DIMS)]

                ob, r_t, term, trunc, _ = env.step(mixed_act)
                done = term or trunc

                s_t1 = np.hstack(ob)

                self.buff.add(s_t, a_t[0], r_t, s_t1, done) # Add to replay buffer

                # Do the batch update
                batch = self.buff.get_batch(self.batch_size)
                states = np.asarray([e[0] for e in batch], dtype=np.float32)
                actions = np.asarray([e[1] for e in batch], dtype=np.float32)
                rewards = np.asarray([e[2] for e in batch], dtype=np.float32)
                new_states = np.asarray([e[3] for e in batch], dtype=np.float32)
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch], dtype=np.float32)

                target_q_values = self.critic.target_model.predict(torch.from_numpy(new_states), self.actor.target_model.predict(torch.from_numpy(new_states))).detach().numpy()

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA * target_q_values[k]
                
                self.critic.model.train_on_batch(torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(y_t))
                a_for_grad = self.actor.model.predict(torch.from_numpy(states))
                grads = self.critic.gradients(states, a_for_grad)
                self.actor.train(states, grads)
                self.actor.target_train()
                self.critic.target_train()
                
                total_reward += r_t
                s_t = s_t1

                # Control prior mixing term
                if j_iter > 0 and i_episode > 50:
                    lambda_track = lambda_max * (1 - np.exp(-factor * np.abs(r_t + GAMMA * np.mean(target_q_values[-1] - base_q[-1]))))
                    lambda_track = np.squeeze(lambda_track)
                else:
                    lambda_track = 10.0
                lambda_store[j_iter] = lambda_track
                base_q = copy.deepcopy(target_q_values)
                
                if done:
                    break
            else:
                raise AssertionError("\"max_steps\" has been reached.")
            
            self.lambda_mix = np.mean(lambda_store) # TODO: lambda mix affected by max_steps?

            logging.info(f"Total Reward {total_reward}, {self.BEST_VAL_NAME} {ob[self.BEST_VAL_IND]}, Last State {ob}, Lambda Mix {self.lambda_mix}")
            logging.info("")
        
        env.close() # This is for shutting down the environment
        logging.info("Finish")
        return None

    def collect_data(self, controllers, tree=False):
        GAMMA = 0.99
        EXPLORE = 200.0 * self.MAX_EPISODE_LEN
        max_steps = 2 * self.MAX_EPISODE_LEN
        done = False
        epsilon = 1
        min_epsilon = 0.01

        if not tree:
            action_prog = controllers[0]
        
        # Generate an environment
        env = gym.make(self.ENV_NAME)

        window = 5

        logging.info(f"{self.ENV_NAME} collection start with Lambda {self.lambda_mix}")
        ob, _ = env.reset()

        s_t = np.hstack(ob)

        total_reward = 0.0
        temp_obs = [[ob[i]] for i in range(len(ob))] + [[0]]
        window_list = [temp_obs[:] for _ in range(window)]

        observation_list = []
        actions_list = []

        for _ in range(max_steps):
            if tree:
                tree_obs = [sensor for obs in temp_obs[:-1] for sensor in obs]
                act_tree = controllers.predict([tree_obs])
                action_action = utils.clip_to_range(act_tree[0][0], self.ACTION_MIN[0], self.ACTION_MAX[0])
            else:
                action_action = utils.clip_to_range(action_prog.pid_execute(window_list), self.ACTION_MIN[0], self.ACTION_MAX[0])
            
            action_prior = [action_action]

            temp_obs = [[ob[i]] for i in range(len(ob))] + [action_prior]
            window_list.pop(0)
            window_list.append(temp_obs[:])

            epsilon -= 1.0 / EXPLORE
            epsilon = max(epsilon, min_epsilon)
            a_t = self.actor.model.predict(torch.from_numpy(s_t.reshape(1, len(s_t)))).detach().numpy()
            mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(self.ACTION_DIMS)]

            if tree:
                new_obs = [item for sublist in temp_obs[:-1] for item in sublist]
                observation_list.append(new_obs[:])
            else:
                observation_list.append(window_list[:])
            actions_list.append(mixed_act[:])

            ob, r_t, term, trunc, _ = env.step(mixed_act)
            done = term or trunc

            s_t1 = np.hstack(ob)

            self.buff.add(s_t, a_t[0], r_t, s_t1, done) # Add to replay buffer

            # Do the batch update
            batch = self.buff.get_batch(self.batch_size)
            rewards = np.asarray([e[2] for e in batch], dtype=np.float32)
            new_states = np.asarray([e[3] for e in batch], dtype=np.float32)
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch], dtype=np.float32)

            target_q_values = self.critic.target_model.predict(torch.from_numpy(new_states), self.actor.target_model.predict(torch.from_numpy(new_states))).detach().numpy()

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]
            
            total_reward += r_t
            s_t = s_t1
            
            if done:
                break
        else:
            raise AssertionError("\"max_steps\" has been reached.")
        
        logging.info(f"Total Reward {total_reward}, {self.BEST_VAL_NAME} {ob[self.BEST_VAL_IND]}, Last State {ob}")
        logging.info("")
        
        env.close()
        
        return observation_list, actions_list

    def label_data(self, controllers, observation_list, tree=False):
        if not tree:
            action_prog = controllers[0]
        actions_list = []
        net_obs_list = []
        logging.info(f"Data labeling started with Lambda {self.lambda_mix}")
        for window_list in observation_list:
            if tree:
                act_tree = controllers.predict([window_list])
                action_action = utils.clip_to_range(act_tree[0][0], self.ACTION_MIN[0], self.ACTION_MAX[0])
                net_obs_list.append(window_list)
            else:
                action_action = utils.clip_to_range(action_prog.pid_execute(window_list), self.ACTION_MIN[0], self.ACTION_MAX[0])
                net_obs = [sensor for obs in window_list[-1] for sensor in obs]
                net_obs_list.append(net_obs[:self.STATE_DIMS])
            
            action_prior = [action_action]

            s_t = np.hstack([[net_obs[:self.STATE_DIMS]]])
            a_t = self.actor.model.predict(s_t.reshape(1, self.STATE_DIMS))
            mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(self.ACTION_DIMS)]

            actions_list.append(mixed_act[:])
        
        return net_obs_list, observation_list, actions_list
