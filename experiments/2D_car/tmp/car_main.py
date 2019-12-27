import numpy as np
import os
import shutil
import asyncio

from config_env import ConfigEnv
from car_env import CarEnv
from car_wsclient import WsClient

#Friquently changed constants
# MAX_EPISODES = 200
# MAX_EP_STEPS = 300
# MAX_EPISODES = 500
# MAX_EP_STEPS = 600

#Train constants
NEED_SAVE = True
LOAD = False

#Eval constants
# NEED_SAVE = False
# LOAD = True

TRAIN_LOOP = {"state": "start"}

state = None
reward = 0
terminal = False
action = None
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
    global reward
    global terminal
    global step_count
    step_count = 0
    state = env.reset()
    reward = 0.0
    terminal = False
    env.render()
    return "send_state"

async def send_state_handler():
    global terminal
    message = "state:"
    for num in state:
        message += str(num) + ','
    message += str(reward) + ','
    if terminal:
        message += '-1' + ','
    else:
        message += '0' + ','
    print("send  %s" % str(message[:-1]))
    ws_client.action_ready = False
    ws_client.send(message[:-1])
    return "step_count"

async def wait_action_handler():
    if ws_client.action_ready:
        return "step_with_action"
    else:
        return "wait_action"

async def step_with_action_handler():
    global state
    global reward
    global terminal
    state,reward,terminal = env.step(ws_client.action)
    env.render()
    return "send_state"

async def step_count_handler():
    global step_count
    step_count += 1
    if step_count > config.MAX_EP_STEPS or terminal:
        print("-------------------------------step_count %s" % str(step_count))
        return "ep_count"
    else:
        return "wait_action"

async def ep_count_handler():
    global ep_count
    ep_count += 1
    print("--------------------------------------------------------ep_count %s" % str(ep_count))
    if ep_count < config.MAX_EPISODES:
        return "reset"
    else:
        return "end"

async def stop_handler():
    print("--------stop_handler")
    ws_client.on_close()

async def unknown_handler():
    print("--------unknown_handler")

def state_selector(arg):
    switcher = { 
        "start": start_handler, 
        "reset": reset_handler,
        "send_state": send_state_handler,
        "wait_action": wait_action_handler,
        "step_with_action": step_with_action_handler,
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
    config = ConfigEnv()
    env = CarEnv(config)
    env.set_fps(30)
    state = env.reset()
    action = env.sample_action()
    ws_client = WsClient()

    asyncio.get_event_loop().run_until_complete(train_loop())

