import numpy as np
import asyncio
import websockets

from DDPG import DDPG

n_sensor = 5
action_dim = 1
state_dim = n_sensor
action_bound = [-1, 1]

action_done = False
sample_action = None
env_state = None
env_reward = None
env_done = None

brain = DDPG()

async def init_handler(websocket, arg_str):
    s = env.init()
    env.render()
    message = "init_done:"
    for num in s:
        message += str(num) + ','
    print("send %s" % str(message[:-1]))
    await websocket.send(message[:-1])

async def reset_handler(websocket, arg_str):
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
    print("DDPG server: unknown_handler")


async def notify_client(websocket, message):
    if websocket:
        await asyncio.wait([websocket.send(message)])

async def register(websocket):
    # TBD
    await notify_client(websocket, 'register_done')

async def unregister(websocket):
    # TBD
    await notify_client(websocket,'unregister_done')

async def nn_choose_act():
    # print("DDPG - nn_choose_act_handler")
    global a
    a = actor.choose_action(s)
    a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
    return "env_step"

async def action_done_handler(websocket, arg_str):
    arg_data_str = arg_str.split(',')

    state_str0 = arg_data_str[:state_dim]
    reward_str0 = arg_data_str[state_dim]

    arr_state_str = np.array(state_str0)
    arr_state_float = arr_state_str.astype(np.float)
    env_state = arr_state_float
    reward_float = float(reward_str0)
    env_reward = reward_float


async def mess_handler(websocket, path):
    await register(websocket)
    try:
        async for message in websocket:
            print("DDPG server receive %s" % str(message))
            cmdHandler, message_data = mess_selector(message)
            await cmdHandler(websocket, message_data)
    finally:
        await unregister(websocket)

def mess_selector(message):
    args = message.split(':')
    switcher = { 
        # "init_done": init_done_handler,
        # "reset_done": reset_done_handler,
        "action_done": action_done_handler,
        # "stop_done": stop_done_handler
    }
    return switcher.get(args[0], unknown_handler), args[1]

if __name__ == '__main__':
# DDPG initialization
    brain.init()

    start_server = websockets.serve(mess_handler, "localhost", 9001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()



# =========================================================
