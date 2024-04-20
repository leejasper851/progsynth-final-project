import numpy as np
import gymnasium as gym
import logging
import torch

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer
import utils
import globals as glb

def function_OU(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)[0]

class NeuralAgent():
    def __init__(self):
        BUFFER_SIZE = 50 * glb.MAX_EPISODE_LEN
        TAU = 0.001 # Target network hyperparameters
        LRA = 0.0001 # Learning rate for actor
        LRC = 0.001 # Learning rate for critic
        self.batch_size = 32
        self.lambda_mix = 1.0

        self.actor = ActorNetwork(glb.STATE_DIMS, glb.ACTION_DIMS, TAU, LRA)
        self.critic = CriticNetwork(glb.STATE_DIMS, glb.ACTION_DIMS, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE) # Create replay buffer

    def update_neural(self, controllers, ippg_iter, episode_count=200, tree=False):
        GAMMA = 0.99
        EXPLORE = 50.0 * glb.MAX_EPISODE_LEN
        max_steps = 2 * glb.MAX_EPISODE_LEN
        done = False
        epsilon = 1
        min_epsilon = 0

        if not tree:
            action_prog = controllers[0]
        
        # Generate an environment
        if glb.ENV_WRAPPER is None:
            env = gym.make(glb.ENV_NAME)
        else:
            env = glb.ENV_WRAPPER(gym.make(glb.ENV_NAME))

        window = 5

        logging.info(f"{glb.ENV_NAME} experiment start with Lambda {self.lambda_mix}")

        for i_episode in range(episode_count):
            logging.info(f"Iteration {ippg_iter}, Episode {i_episode}")
            logging.info(f"Replay Buffer {self.buff.count()}")
            logging.info("")
            ob, _ = env.reset()

            s_t = np.hstack(ob)

            total_reward = 0.0
            temp_obs = [[ob[i]] for i in range(len(ob))] + [[0]]
            window_list = [temp_obs[:] for _ in range(window)]

            for _ in range(max_steps):
                if tree:
                    tree_obs = [sensor for obs in temp_obs[:-1] for sensor in obs]
                    act_tree = controllers.predict([tree_obs])
                    action_action = utils.clip_to_range(act_tree[0][0], glb.ACTION_MIN[0], glb.ACTION_MAX[0])
                else:
                    action_action = utils.clip_to_range(action_prog.pid_execute(window_list), glb.ACTION_MIN[0], glb.ACTION_MAX[0])
                
                action_prior = [action_action]

                temp_obs = [[ob[i]] for i in range(len(ob))] + [action_prior]
                window_list.pop(0)
                window_list.append(temp_obs[:])

                epsilon -= 1.0 / EXPLORE
                epsilon = max(epsilon, min_epsilon)
                a_t = np.zeros([1, glb.ACTION_DIMS])
                noise_t = np.zeros([1, glb.ACTION_DIMS])

                a_t_original = self.actor.model.predict(torch.from_numpy(s_t.reshape(1, len(s_t)))).detach().numpy()
                noise_t[0][0] = max(epsilon, 0) * function_OU(a_t_original[0][0], 0.0, 0.15, 0.2)

                a_t[0][0] = a_t_original[0][0] + noise_t[0][0]

                mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(glb.ACTION_DIMS)]

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
                
                if done:
                    break
            else:
                raise AssertionError("\"max_steps\" has been reached.")
            
            self.lambda_mix = 1 + (1 / 50) * (2 ** (0.5 * (ippg_iter + (i_episode + 1) / episode_count)))

            logging.info(f"Total Reward {total_reward}, {glb.BEST_VAL_NAME} {ob[glb.BEST_VAL_IND]}, Last State {ob}, Lambda Mix {self.lambda_mix}")
            logging.info("")
        
        env.close() # This is for shutting down the environment
        logging.info("Finish")
        logging.info("")
        return None

    def collect_data(self, controllers, tree=False):
        GAMMA = 0.99
        max_steps = 2 * glb.MAX_EPISODE_LEN
        done = False

        if not tree:
            action_prog = controllers[0]
        
        # Generate an environment
        if glb.ENV_WRAPPER is None:
            env = gym.make(glb.ENV_NAME)
        else:
            env = glb.ENV_WRAPPER(gym.make(glb.ENV_NAME))

        window = 5

        logging.info(f"{glb.ENV_NAME} collection start with Lambda {self.lambda_mix}")
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
                action_action = utils.clip_to_range(act_tree[0][0], glb.ACTION_MIN[0], glb.ACTION_MAX[0])
            else:
                action_action = utils.clip_to_range(action_prog.pid_execute(window_list), glb.ACTION_MIN[0], glb.ACTION_MAX[0])
            
            action_prior = [action_action]

            temp_obs = [[ob[i]] for i in range(len(ob))] + [action_prior]
            window_list.pop(0)
            window_list.append(temp_obs[:])

            a_t = self.actor.model.predict(torch.from_numpy(s_t.reshape(1, len(s_t)))).detach().numpy()
            mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(glb.ACTION_DIMS)]

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
        
        logging.info(f"Total Reward {total_reward}, {glb.BEST_VAL_NAME} {ob[glb.BEST_VAL_IND]}, Last State {ob}")
        logging.info("")
        
        env.close()
        
        return observation_list, actions_list

    def label_data(self, controllers, observation_list, tree=False):
        if not tree:
            action_prog = controllers[0]
        actions_list = []
        logging.info(f"Data labeling started with Lambda {self.lambda_mix}")
        for window_list in observation_list:
            if tree:
                act_tree = controllers.predict([window_list])
                action_action = utils.clip_to_range(act_tree[0][0], glb.ACTION_MIN[0], glb.ACTION_MAX[0])
            else:
                action_action = utils.clip_to_range(action_prog.pid_execute(window_list), glb.ACTION_MIN[0], glb.ACTION_MAX[0])
                net_obs = [sensor for obs in window_list[-1] for sensor in obs]
            
            action_prior = [action_action]

            s_t = np.hstack([[net_obs[:glb.STATE_DIMS]]], dtype=np.float32)
            a_t = self.actor.model.predict(torch.from_numpy(s_t.reshape(1, glb.STATE_DIMS))).detach().numpy()
            mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(glb.ACTION_DIMS)]

            actions_list.append(mixed_act[:])
        
        return actions_list
