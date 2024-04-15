import numpy as np
import gymnasium as gym
import logging
import torch
import argparse
import sys
import os

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer

class FunctionOU():
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

def run_ddpg(amodel, cmodel, train_indicator=0, seeded=1337):
    OU = FunctionOU()
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001 # Target network hyperparameters
    LRA = 0.0001 # Learning rate for actor
    LRC = 0.001 # Learning rate for critic

    state_dim = 2
    action_dim = 1

    np.random.seed(seeded)

    EXPLORE = 200000.0
    if train_indicator:
        episode_count = 1000
    else:
        episode_count = 5
    max_steps = 20000
    done = False
    step = 0
    epsilon = 1
    min_epsilon = 0.01

    actor = ActorNetwork(state_dim, action_dim, TAU, LRA)
    critic = CriticNetwork(state_dim, action_dim, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE) # Create replay buffer

    # Generate an environment
    env = gym.make("MountainCarContinuous-v0")

    if not train_indicator:
        # Now load the weight
        logging.info("Now we load the weights")
        try:
            actor.model.load_state_dict(amodel)
            critic.model.load_state_dict(cmodel)
            actor.target_model.load_state_dict(amodel)
            critic.target_model.load_state_dict(cmodel)
            logging.info("Weights loaded successfully")
        except:
            logging.info("Cannot find the weights")
            exit()
    
    logging.info("Experiment start")
    best_pos = -1.2
    best_total_reward = -100

    for i_episode in range(episode_count):
        logging.info("Episode : " + str(i_episode) + " Replay Buffer " + str(buff.count()))
        ob, _ = env.reset()

        s_t = np.hstack(ob)

        total_reward = 0.0

        for _ in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            epsilon = max(epsilon, min_epsilon)
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(torch.from_numpy(s_t.reshape(1, len(s_t)))).detach().numpy()
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.15, 0.2)

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
                loss += critic.model.train_on_batch(torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(y_t))
                a_for_grad = actor.model.predict(torch.from_numpy(states))
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()
            
            total_reward += r_t
            s_t = s_t1

            best_pos = max(best_pos, ob[0])
            
            step += 1
            if done:
                break
        
        logging.info("Episode {}, Position {}, Velocity {}".format(i_episode, ob[0], ob[1]))
        best_total_reward = max(best_total_reward, total_reward)

        if train_indicator and i_episode > 20:
            if np.mod(i_episode, 5) == 0:
                logging.info("Now we save the model")
                torch.save(actor.model.state_dict(), "run/ddpg_actor_weights_periodic.pt")
                torch.save(critic.model.state_dict(), "run/ddpg_critic_weights_periodic.pt")
        logging.info("TOTAL REWARD @ " + str(i_episode) + "-th Episode : Reward " + str(total_reward))
        logging.info("Best Position {} Best Total Reward {}".format(best_pos, best_total_reward))
    env.close() # This is for shutting down the environment
    logging.info("Finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None)
    parser.add_argument("--mode", default=1, type=int) # 0 - run, 1 - train
    parser.add_argument("--actormodel", default=None)
    parser.add_argument("--criticmodel", default=None)
    parser.add_argument("--logname", default="MountainCarDDPG")
    args = parser.parse_args()

    os.makedirs("run/", exist_ok=True)

    log_path = "run"
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
