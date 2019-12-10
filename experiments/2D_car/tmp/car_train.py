import numpy as np
import os
import shutil
import asyncio

from car_env import CarEnv
from car_client import RemoteNnClient

#Friquently changed constants
MAX_EPISODES = 100
MAX_EP_STEPS = 500
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

state = None
action = None
step_done = False
ws_client = None
step_count = 0
ep_count = 0



async def start_handler():
    print("---start_handler")
    env.init()
    return "reset"

async def reset_handler():
    print("---reset_handler")
    global ws_client
    global state
    state = env.reset()
    env.render()
    return "send_state"

async def send_state_handler():
    print("---send_state_handler")
    message = "state:"
    for num in state:
        message += str(num) + ','
    print("send %s" % str(message[:-1]))
    ws_client.action_ready = False
    ws_client.send(message[:-1])
    return "wait_action"

async def wait_action_handler():
    # print("---wait_action_handler")
    if ws_client.action_ready:
        print("---wait_action_handler ws_client.action_ready == True")
        return "start_step_with_action"
    else:
        return "wait_action"

async def start_step_with_action_handler():
    print("---start_step_with_action_handler")
    global step_done
    step_done = False
    env.step(action)
    step_done = True
    return "wait_step_done"

async def wait_step_done_handler():
    print("---wait_step_done_handler")
    if step_done:
        return "get_state"
    else:
        return "wait_step_done"

async def get_state_handler():
    print("---get_state_handler")
    global state
    state = env._get_state()
    return "step_count"

# async def calc_reward_handler():
#     reward = env.calc_reward()
#     return "step_count"

async def step_count_handler():
    print("---step_count_handler")
    global step_count
    step_count += 1
    print("-------------------------------step_count %s" % str(step_count))
    if step_count < MAX_EP_STEPS:
        return "send_state"
    else:
        return "ep_count"

async def ep_count_handler():
    print("---ep_count_handler")
    global ep_count
    ep_count += 1
    print("--------------------------------------------------------ep_count %s" % str(ep_count))
    if ep_count < MAX_EPISODES:
        return "send_state"
    else:
        return "stop"


# async def step_handler(websocket, arg_str):
#     arg_data_str = arg_str.split(',')
#     arr_str = np.array(arg_data_str)
#     arr_float = arr_str.astype(np.float)
#     s, r, terminal = env.step(arr_float)
#     env.render()

#     message = "step_done:"
#     for num in s:
#         message += str(num) + ','
#     message += str(r) + ','
#     message += str(terminal)
#     print("send %s" % str(message))
#     await websocket.send(message)

async def stop_handler(websocket, arg_str):
    print("--------stop_handler")
    # env.stop()
    # message = "stop_done:0"
    # print("send %s" % str(message))
    # await websocket.send(message)

async def unknown_handler():
    print("--------unknown_handler")


async def mess_handler(websocket, path):
    print("---mess_handler")
    async for message in websocket:
        # print("--------for message in websocket")
        print("receive %s" % str(message))
        cmdHandler, message_data = command_selector(message)
        await cmdHandler(websocket, message_data)

def state_selector(arg):
    # print("---state_selector")
    switcher = { 
        "start": start_handler, 
        "reset": reset_handler,
        # "get_state_reward": get_state_reward_handler,
        "send_state": send_state_handler,
        "wait_action": wait_action_handler,
        "start_step_with_action": start_step_with_action_handler,
        "wait_step_done": wait_step_done_handler,
        "get_state": get_state_handler,
        # "calc_reward": calc_reward_handler,
        "step_count": step_count_handler,
        "ep_count": ep_count_handler,
        "stop": stop_handler,
    } 
    return switcher.get(arg, unknown_handler)


async def train_loop():
    print("---train_loop")
    need_do_save = NEED_SAVE
    continue_train_loop = True
    while (TRAIN_LOOP["state"] != "end") & continue_train_loop:
        continue_train_loop = True
        tr_state = TRAIN_LOOP["state"]
        stateHandler = state_selector(tr_state)
        new_tr_state = await stateHandler()
        TRAIN_LOOP["state"] = new_tr_state
    print("train_loop end")


if __name__ == '__main__':
    env = CarEnv()
    env.set_fps(30)
    state = env.reset()
    action = env.sample_action()
    ws_client = RemoteNnClient()

    asyncio.get_event_loop().run_until_complete(train_loop())

