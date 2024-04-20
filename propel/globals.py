from pendulum import PendulumThetaEnv

ENV_NAME = "Pendulum-v1"
ENV_WRAPPER = PendulumThetaEnv
STATE_DIMS = 2
ACTION_DIMS = 1
MAX_EPISODE_LEN = 200
ACTION_MIN = (-2,)
ACTION_MAX = (2,)
BEST_VAL_IND = 0
BEST_VAL_MAX = True
BEST_VAL_NAME = "Theta"
