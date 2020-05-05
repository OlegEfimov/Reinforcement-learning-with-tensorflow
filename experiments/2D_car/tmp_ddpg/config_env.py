
class ConfigEnv(object):
  MAX_EPISODES = 10
  MAX_EP_STEPS = 100
  # MAX_EPISODES = 200
  # MAX_EP_STEPS = 300

  # MAX_EPISODES = 500
  # MAX_EP_STEPS = 600

  LR_A = 1e-4  # learning rate for actor
  LR_C = 1e-4  # learning rate for critic
  GAMMA = 0.9  # reward discount
  REPLACE_ITER_A = 800
  REPLACE_ITER_C = 700
  MEMORY_CAPACITY = 2000
  BATCH_SIZE = 16
  VAR_MIN = 0.1
  RENDER = True
  LOAD = False
  DISCRETE_ACTION = False

  STATE_DIM = 10
  ACTION_DIM = 2
  ACTION_BOUND = [-1.0, 1.0]
  ACTION_RANGE = [1.0]
