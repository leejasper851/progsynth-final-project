import numpy as np
import gymnasium as gym
import logging
import torch
import argparse
import sys
import os
import matplotlib.pyplot as plt

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer
from pendulum import PendulumThetaEnv

ENV_NAME = "Pendulum-v1" #TODO: move global variables to config file
STATE_DIMS = 2
ACTION_DIMS = 1
MAX_EPISODE_LEN = 200
BEST_VAL_IND = 0
BEST_VAL_MAX = True
BEST_VAL_NAME = "Theta"

def function_OU(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)[0]

def run_ddpg(amodel, cmodel, train_indicator=0, seeded=1337):
    BUFFER_SIZE = 50 * MAX_EPISODE_LEN
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001 # Target network hyperparameters
    LRA = 0.0001 # Learning rate for actor
    LRC = 0.001 # Learning rate for critic

    np.random.seed(seeded)

    EXPLORE = 50.0 * MAX_EPISODE_LEN
    if train_indicator:
        episode_count = 600
    else:
        episode_count = 1000
    max_steps = 2 * MAX_EPISODE_LEN
    epsilon = 1
    min_epsilon = 0

    actor = ActorNetwork(STATE_DIMS, ACTION_DIMS, TAU, LRA)
    critic = CriticNetwork(STATE_DIMS, ACTION_DIMS, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE) # Create replay buffer

    # Generate an environment
    env = PendulumThetaEnv(gym.make(ENV_NAME)) #TODO: make more general everywhere

    if not train_indicator:
        # Now load the weight
        logging.info("Now we load the weights")
        try:
            actor.model.load_state_dict(torch.load(amodel))
            critic.model.load_state_dict(torch.load(cmodel))
            actor.target_model.load_state_dict(torch.load(amodel))
            critic.target_model.load_state_dict(torch.load(cmodel))
            logging.info("Weights loaded successfully")
        except:
            logging.info("Cannot find the weights")
            exit()
    
    logging.info(f"{ENV_NAME} experiment start")
    best_val = float("-inf") if BEST_VAL_MAX else float("inf")
    best_total_reward = float("-inf")
    avg_total_reward = 0

    logging.info("")

    plot_x = []
    plot_y = []

    for i_episode in range(episode_count):
        logging.info(f"Episode {i_episode}")
        logging.info(f"Replay Buffer {buff.count()}")
        logging.info("")
        ob, _ = env.reset()

        s_t = np.hstack(ob)

        total_reward = 0.0

        for _ in range(max_steps):
            epsilon -= 1.0 / EXPLORE
            epsilon = max(epsilon, min_epsilon)
            a_t = np.zeros([1, ACTION_DIMS])
            noise_t = np.zeros([1, ACTION_DIMS])

            a_t_original = actor.model.predict(torch.from_numpy(s_t.reshape(1, len(s_t)))).detach().numpy()
            noise_t[0][0] = train_indicator * max(epsilon, 0) * function_OU(a_t_original[0][0], 0.0, 0.15, 0.2)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]

            ob, r_t, term, trunc, _ = env.step(a_t[0])
            done = term or trunc

            s_t1 = np.hstack(ob)

            buff.add(s_t, a_t[0], r_t, s_t1, done) # Add to replay buffer

            # Do the batch update
            batch = buff.get_batch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch], dtype=np.float32)
            actions = np.asarray([e[1] for e in batch], dtype=np.float32)
            rewards = np.asarray([e[2] for e in batch], dtype=np.float32)
            new_states = np.asarray([e[3] for e in batch], dtype=np.float32)
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch], dtype=np.float32)

            target_q_values = critic.target_model.predict(torch.from_numpy(new_states), actor.target_model.predict(torch.from_numpy(new_states))).detach().numpy()

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]
            
            if train_indicator:
                critic.model.train_on_batch(torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(y_t))
                a_for_grad = actor.model.predict(torch.from_numpy(states))
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()
            
            total_reward += r_t
            s_t = s_t1

            best_val = max(best_val, ob[BEST_VAL_IND]) if BEST_VAL_MAX else min(best_val, ob[BEST_VAL_IND])
            
            if done:
                break
        else:
            raise AssertionError("\"max_steps\" has been reached.")
        
        best_total_reward = max(best_total_reward, total_reward)
        avg_total_reward += total_reward

        plot_x.append(i_episode)
        plot_y.append(total_reward)

        logging.info(f"Total Reward {total_reward}, {BEST_VAL_NAME} {ob[BEST_VAL_IND]}, Last State {ob}")
        logging.info(f"Best Total Reward {best_total_reward}, Best {BEST_VAL_NAME} {best_val}")

        if train_indicator and i_episode > 20 and i_episode % 5 == 0:
            logging.info("Now we save the model")
            torch.save(actor.model.state_dict(), "run_ddpg/ddpg_actor_weights_periodic.pt")
            torch.save(critic.model.state_dict(), "run_ddpg/ddpg_critic_weights_periodic.pt")
        
        logging.info("")
    
    env.close() # This is for shutting down the environment
    avg_total_reward /= episode_count
    logging.info(f"Average Total Reward {avg_total_reward} (over {episode_count} episodes)")
    logging.info("Finish")
    logging.info("")

    if train_indicator:
        plt.plot(plot_x, plot_y)
        plt.title(f"{ENV_NAME} DDPG Training")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig("run_ddpg/training_plot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None)
    parser.add_argument("--mode", default=1, type=int) # 0 - run, 1 - train
    parser.add_argument("--actormodel", default="run_ddpg_final/ddpg_actor_weights_periodic.pt")
    parser.add_argument("--criticmodel", default="run_ddpg_final/ddpg_critic_weights_periodic.pt")
    parser.add_argument("--logname", default="DDPG")
    args = parser.parse_args()

    os.makedirs("run_ddpg/", exist_ok=True)

    log_path = "run_ddpg"
    log_filename = args.logname
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(log_path, log_filename)),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")

    run_ddpg(args.actormodel, args.criticmodel, train_indicator=args.mode, seeded=args.seed)
