import numpy as np
import os
import shutil
import asyncio

from car_env import CarEnv
from car_client import RemoteNnClient

#Friquently changed constants
MAX_EPISODES = 3
MAX_EP_STEPS = 10
# MAX_EPISODES = 500
# MAX_EP_STEPS = 600

MEMORY_CAPACITY = 2000

#Train constants
# NEED_SAVE = True
# LOAD = False

#Eval constants
NEED_SAVE = False
LOAD = True

TRAIN_LOOP = {"state": "start"}


async def start_handler():
    # print("DDPG - start_handler")
    env.init()
    return "wait_init_done"

async def reset_handler():
    s = env.reset()
    env.render()
    message = "reset_done:"
    for num in s:
        message += str(num) + ','
    print("send %s" % str(message[:-1]))
    await websocket.send(message[:-1])

async def step_handler(websocket, arg_str):
    arg_data_str = arg_str.split(',')
    arr_str = np.array(arg_data_str)
    arr_float = arr_str.astype(np.float)
    s, r, terminal = env.step(arr_float)
    env.render()

    message = "step_done:"
    for num in s:
        message += str(num) + ','
    message += str(r) + ','
    message += str(terminal)
    print("send %s" % str(message))
    await websocket.send(message)

async def stop_handler(websocket, arg_str):
    # print("--------stop_handler")
    env.stop()
    message = "stop_done:0"
    print("send %s" % str(message))
    await websocket.send(message)

async def unknown_handler():
    print("--------unknown_handler")


async def mess_handler(websocket, path):
    async for message in websocket:
        # print("--------for message in websocket")
        print("receive %s" % str(message))
        cmdHandler, message_data = command_selector(message)
        await cmdHandler(websocket, message_data)

def state_selector(message): 
    args = message.split(':')
    switcher = { 
        "start": start_handler, 
        # "wait_init_done": wait_init_done_handler,
        "start_episode": start_episode_handler,
        "stop_episode": stop_episode_handler,
        # "send_reset": send_reset_handler,
        # "wait_reset_done": wait_reset_done_handler,
        "start_step": start_step_handler,
        "stop_step": stop_step_handler,
        # "nn_choose_act": nn_choose_act_handler,
        # "env_step": env_step_handler,
        # "wait_step_done": wait_step_done_handler,
        "nn_learn": nn_learn_handler,
        # "stop": stop_handler,
        # "wait_stop_done": wait_stop_done_handler,

        "init": init_handler,
        "init": init_handler,
        "reset": reset_handler,
        "step": step_handler,
        "stop": stop_handler
    } 
    return switcher.get(args[0], unknown_handler), args[1]


async def train_loop():
    need_do_save = NEED_SAVE
    continue_train_loop = True
    while (TRAIN_LOOP["state"] != "end") & continue_train_loop:
        continue_train_loop = True
        tr_state = TRAIN_LOOP["state"]
        stateHandler = state_selector(tr_state)
        new_tr_state = await stateHandler()
        TRAIN_LOOP["state"] = new_tr_state


if __name__ == '__main__':
    # np.random.seed(1)
    tf.compat.v1.set_random_seed(1)
    env = CarEnv()
    env.set_fps(30)
    s = env.reset()
    action = env.sample_action()
    ws_client = RemoteNnClient()

    asyncio.get_event_loop().run_until_complete(train_loop())

